package blazen

// #include <blazen.h>
import "C"

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"reflect"
	"runtime"
	"runtime/cgo"
	"sync"
	"sync/atomic"
	"unsafe"
)

// This is needed, because as of go 1.24
// type RustBuffer C.RustBuffer cannot have methods,
// RustBuffer is treated as non-local type
type GoRustBuffer struct {
	inner C.RustBuffer
}

type RustBufferI interface {
	AsReader() *bytes.Reader
	Free()
	ToGoBytes() []byte
	Data() unsafe.Pointer
	Len() uint64
	Capacity() uint64
}

// C.RustBuffer fields exposed as an interface so they can be accessed in different Go packages.
// See https://github.com/golang/go/issues/13467
type ExternalCRustBuffer interface {
	Data() unsafe.Pointer
	Len() uint64
	Capacity() uint64
}

func RustBufferFromC(b C.RustBuffer) ExternalCRustBuffer {
	return GoRustBuffer{
		inner: b,
	}
}

func CFromRustBuffer(b ExternalCRustBuffer) C.RustBuffer {
	return C.RustBuffer{
		capacity: C.uint64_t(b.Capacity()),
		len:      C.uint64_t(b.Len()),
		data:     (*C.uchar)(b.Data()),
	}
}

func RustBufferFromExternal(b ExternalCRustBuffer) GoRustBuffer {
	return GoRustBuffer{
		inner: C.RustBuffer{
			capacity: C.uint64_t(b.Capacity()),
			len:      C.uint64_t(b.Len()),
			data:     (*C.uchar)(b.Data()),
		},
	}
}

func (cb GoRustBuffer) Capacity() uint64 {
	return uint64(cb.inner.capacity)
}

func (cb GoRustBuffer) Len() uint64 {
	return uint64(cb.inner.len)
}

func (cb GoRustBuffer) Data() unsafe.Pointer {
	return unsafe.Pointer(cb.inner.data)
}

func (cb GoRustBuffer) AsReader() *bytes.Reader {
	b := unsafe.Slice((*byte)(cb.inner.data), C.uint64_t(cb.inner.len))
	return bytes.NewReader(b)
}

func (cb GoRustBuffer) Free() {
	rustCall(func(status *C.RustCallStatus) bool {
		C.ffi_blazen_uniffi_rustbuffer_free(cb.inner, status)
		return false
	})
}

func (cb GoRustBuffer) ToGoBytes() []byte {
	return C.GoBytes(unsafe.Pointer(cb.inner.data), C.int(cb.inner.len))
}

func stringToRustBuffer(str string) C.RustBuffer {
	return bytesToRustBuffer([]byte(str))
}

func bytesToRustBuffer(b []byte) C.RustBuffer {
	if len(b) == 0 {
		return C.RustBuffer{}
	}
	// We can pass the pointer along here, as it is pinned
	// for the duration of this call
	foreign := C.ForeignBytes{
		len:  C.int(len(b)),
		data: (*C.uchar)(unsafe.Pointer(&b[0])),
	}

	return rustCall(func(status *C.RustCallStatus) C.RustBuffer {
		return C.ffi_blazen_uniffi_rustbuffer_from_bytes(foreign, status)
	})
}

type BufLifter[GoType any] interface {
	Lift(value RustBufferI) GoType
}

type BufLowerer[GoType any] interface {
	Lower(value GoType) C.RustBuffer
}

type BufReader[GoType any] interface {
	Read(reader io.Reader) GoType
}

type BufWriter[GoType any] interface {
	Write(writer io.Writer, value GoType)
}

func LowerIntoRustBuffer[GoType any](bufWriter BufWriter[GoType], value GoType) C.RustBuffer {
	// This might be not the most efficient way but it does not require knowing allocation size
	// beforehand
	var buffer bytes.Buffer
	bufWriter.Write(&buffer, value)

	bytes, err := io.ReadAll(&buffer)
	if err != nil {
		panic(fmt.Errorf("reading written data: %w", err))
	}
	return bytesToRustBuffer(bytes)
}

func LiftFromRustBuffer[GoType any](bufReader BufReader[GoType], rbuf RustBufferI) GoType {
	defer rbuf.Free()
	reader := rbuf.AsReader()
	item := bufReader.Read(reader)
	if reader.Len() > 0 {
		// TODO: Remove this
		leftover, _ := io.ReadAll(reader)
		panic(fmt.Errorf("Junk remaining in buffer after lifting: %s", string(leftover)))
	}
	return item
}

func rustCallWithError[E any, U any](converter BufReader[E], callback func(*C.RustCallStatus) U) (U, E) {
	var status C.RustCallStatus
	returnValue := callback(&status)
	err := checkCallStatus(converter, status)
	return returnValue, err
}

func checkCallStatus[E any](converter BufReader[E], status C.RustCallStatus) E {
	switch status.code {
	case 0:
		var zero E
		return zero
	case 1:
		return LiftFromRustBuffer(converter, GoRustBuffer{inner: status.errorBuf})
	case 2:
		// when the rust code sees a panic, it tries to construct a rustBuffer
		// with the message.  but if that code panics, then it just sends back
		// an empty buffer.
		if status.errorBuf.len > 0 {
			panic(fmt.Errorf("%s", FfiConverterStringINSTANCE.Lift(GoRustBuffer{inner: status.errorBuf})))
		} else {
			panic(fmt.Errorf("Rust panicked while handling Rust panic"))
		}
	default:
		panic(fmt.Errorf("unknown status code: %d", status.code))
	}
}

func checkCallStatusUnknown(status C.RustCallStatus) error {
	switch status.code {
	case 0:
		return nil
	case 1:
		panic(fmt.Errorf("function not returning an error returned an error"))
	case 2:
		// when the rust code sees a panic, it tries to construct a C.RustBuffer
		// with the message.  but if that code panics, then it just sends back
		// an empty buffer.
		if status.errorBuf.len > 0 {
			panic(fmt.Errorf("%s", FfiConverterStringINSTANCE.Lift(GoRustBuffer{
				inner: status.errorBuf,
			})))
		} else {
			panic(fmt.Errorf("Rust panicked while handling Rust panic"))
		}
	default:
		return fmt.Errorf("unknown status code: %d", status.code)
	}
}

func rustCall[U any](callback func(*C.RustCallStatus) U) U {
	returnValue, err := rustCallWithError[error](nil, callback)
	if err != nil {
		panic(err)
	}
	return returnValue
}

type NativeError interface {
	AsError() error
}

func writeInt8(writer io.Writer, value int8) {
	if err := binary.Write(writer, binary.BigEndian, value); err != nil {
		panic(err)
	}
}

func writeUint8(writer io.Writer, value uint8) {
	if err := binary.Write(writer, binary.BigEndian, value); err != nil {
		panic(err)
	}
}

func writeInt16(writer io.Writer, value int16) {
	if err := binary.Write(writer, binary.BigEndian, value); err != nil {
		panic(err)
	}
}

func writeUint16(writer io.Writer, value uint16) {
	if err := binary.Write(writer, binary.BigEndian, value); err != nil {
		panic(err)
	}
}

func writeInt32(writer io.Writer, value int32) {
	if err := binary.Write(writer, binary.BigEndian, value); err != nil {
		panic(err)
	}
}

func writeUint32(writer io.Writer, value uint32) {
	if err := binary.Write(writer, binary.BigEndian, value); err != nil {
		panic(err)
	}
}

func writeInt64(writer io.Writer, value int64) {
	if err := binary.Write(writer, binary.BigEndian, value); err != nil {
		panic(err)
	}
}

func writeUint64(writer io.Writer, value uint64) {
	if err := binary.Write(writer, binary.BigEndian, value); err != nil {
		panic(err)
	}
}

func writeFloat32(writer io.Writer, value float32) {
	if err := binary.Write(writer, binary.BigEndian, value); err != nil {
		panic(err)
	}
}

func writeFloat64(writer io.Writer, value float64) {
	if err := binary.Write(writer, binary.BigEndian, value); err != nil {
		panic(err)
	}
}

func readInt8(reader io.Reader) int8 {
	var result int8
	if err := binary.Read(reader, binary.BigEndian, &result); err != nil {
		panic(err)
	}
	return result
}

func readUint8(reader io.Reader) uint8 {
	var result uint8
	if err := binary.Read(reader, binary.BigEndian, &result); err != nil {
		panic(err)
	}
	return result
}

func readInt16(reader io.Reader) int16 {
	var result int16
	if err := binary.Read(reader, binary.BigEndian, &result); err != nil {
		panic(err)
	}
	return result
}

func readUint16(reader io.Reader) uint16 {
	var result uint16
	if err := binary.Read(reader, binary.BigEndian, &result); err != nil {
		panic(err)
	}
	return result
}

func readInt32(reader io.Reader) int32 {
	var result int32
	if err := binary.Read(reader, binary.BigEndian, &result); err != nil {
		panic(err)
	}
	return result
}

func readUint32(reader io.Reader) uint32 {
	var result uint32
	if err := binary.Read(reader, binary.BigEndian, &result); err != nil {
		panic(err)
	}
	return result
}

func readInt64(reader io.Reader) int64 {
	var result int64
	if err := binary.Read(reader, binary.BigEndian, &result); err != nil {
		panic(err)
	}
	return result
}

func readUint64(reader io.Reader) uint64 {
	var result uint64
	if err := binary.Read(reader, binary.BigEndian, &result); err != nil {
		panic(err)
	}
	return result
}

func readFloat32(reader io.Reader) float32 {
	var result float32
	if err := binary.Read(reader, binary.BigEndian, &result); err != nil {
		panic(err)
	}
	return result
}

func readFloat64(reader io.Reader) float64 {
	var result float64
	if err := binary.Read(reader, binary.BigEndian, &result); err != nil {
		panic(err)
	}
	return result
}

func init() {

	FfiConverterCompletionStreamSinkINSTANCE.register()
	FfiConverterCustomProviderINSTANCE.register()
	FfiConverterStepHandlerINSTANCE.register()
	FfiConverterToolHandlerINSTANCE.register()
	uniffiCheckChecksums()
}

func uniffiCheckChecksums() {
	// Get the bindings contract version from our ComponentInterface
	bindingsContractVersion := 30
	// Get the scaffolding contract version by calling the into the dylib
	scaffoldingContractVersion := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint32_t {
		return C.ffi_blazen_uniffi_uniffi_contract_version()
	})
	if bindingsContractVersion != int(scaffoldingContractVersion) {
		// If this happens try cleaning and rebuilding your project
		panic("blazen: UniFFI contract version mismatch")
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_version()
		})
		if checksum != 61949 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_version: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_complete_batch()
		})
		if checksum != 21646 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_complete_batch: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_complete_batch_blocking()
		})
		if checksum != 63435 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_complete_batch_blocking: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_diffusion_model()
		})
		if checksum != 18747 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_diffusion_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_fal_image_gen_model()
		})
		if checksum != 23891 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_fal_image_gen_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_fal_stt_model()
		})
		if checksum != 31656 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_fal_stt_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_fal_tts_model()
		})
		if checksum != 32558 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_fal_tts_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_piper_tts_model()
		})
		if checksum != 57207 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_piper_tts_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_whisper_stt_model()
		})
		if checksum != 40916 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_whisper_stt_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_redb_checkpoint_store()
		})
		if checksum != 15901 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_redb_checkpoint_store: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_valkey_checkpoint_store()
		})
		if checksum != 24389 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_valkey_checkpoint_store: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_custom_provider_from_foreign()
		})
		if checksum != 38880 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_custom_provider_from_foreign: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_lm_studio()
		})
		if checksum != 15777 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_lm_studio: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_openai_compat_config()
		})
		if checksum != 41561 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_openai_compat_config: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_ollama()
		})
		if checksum != 53366 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_ollama: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_openai_compat()
		})
		if checksum != 1651 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_openai_compat: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_anthropic_completion_model()
		})
		if checksum != 51033 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_anthropic_completion_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_azure_completion_model()
		})
		if checksum != 44043 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_azure_completion_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_bedrock_completion_model()
		})
		if checksum != 8513 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_bedrock_completion_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_candle_completion_model()
		})
		if checksum != 50089 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_candle_completion_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_candle_embedding_model()
		})
		if checksum != 49772 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_candle_embedding_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_cohere_completion_model()
		})
		if checksum != 22601 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_cohere_completion_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_custom_completion_model_with_openai_protocol()
		})
		if checksum != 52541 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_custom_completion_model_with_openai_protocol: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_deepseek_completion_model()
		})
		if checksum != 51214 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_deepseek_completion_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_fal_completion_model()
		})
		if checksum != 56790 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_fal_completion_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_fal_embedding_model()
		})
		if checksum != 50719 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_fal_embedding_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_fastembed_embedding_model()
		})
		if checksum != 27141 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_fastembed_embedding_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_fireworks_completion_model()
		})
		if checksum != 60773 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_fireworks_completion_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_gemini_completion_model()
		})
		if checksum != 4509 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_gemini_completion_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_groq_completion_model()
		})
		if checksum != 2438 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_groq_completion_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_llamacpp_completion_model()
		})
		if checksum != 2285 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_llamacpp_completion_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_lm_studio_completion_model()
		})
		if checksum != 21338 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_lm_studio_completion_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_mistral_completion_model()
		})
		if checksum != 22583 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_mistral_completion_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_mistralrs_completion_model()
		})
		if checksum != 26159 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_mistralrs_completion_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_ollama_completion_model()
		})
		if checksum != 4717 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_ollama_completion_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_openai_compat_completion_model()
		})
		if checksum != 63435 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_openai_compat_completion_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_openai_completion_model()
		})
		if checksum != 45922 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_openai_completion_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_openai_embedding_model()
		})
		if checksum != 64561 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_openai_embedding_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_openrouter_completion_model()
		})
		if checksum != 43812 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_openrouter_completion_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_perplexity_completion_model()
		})
		if checksum != 57728 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_perplexity_completion_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_together_completion_model()
		})
		if checksum != 20330 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_together_completion_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_tract_embedding_model()
		})
		if checksum != 32866 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_tract_embedding_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_xai_completion_model()
		})
		if checksum != 397 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_xai_completion_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_init()
		})
		if checksum != 33802 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_init: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_complete_streaming()
		})
		if checksum != 3202 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_complete_streaming: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_complete_streaming_blocking()
		})
		if checksum != 29923 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_complete_streaming_blocking: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_init_langfuse()
		})
		if checksum != 16496 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_init_langfuse: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_init_otlp()
		})
		if checksum != 38653 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_init_otlp: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_init_prometheus()
		})
		if checksum != 34029 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_init_prometheus: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_parse_workflow_history()
		})
		if checksum != 59919 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_parse_workflow_history: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_shutdown_telemetry()
		})
		if checksum != 49951 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_shutdown_telemetry: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_agent_run()
		})
		if checksum != 39677 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_agent_run: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_agent_run_blocking()
		})
		if checksum != 6800 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_agent_run_blocking: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_toolhandler_execute()
		})
		if checksum != 52993 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_toolhandler_execute: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_imagegenmodel_generate()
		})
		if checksum != 52613 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_imagegenmodel_generate: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_imagegenmodel_generate_blocking()
		})
		if checksum != 37612 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_imagegenmodel_generate_blocking: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_sttmodel_transcribe()
		})
		if checksum != 4106 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_sttmodel_transcribe: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_sttmodel_transcribe_blocking()
		})
		if checksum != 20646 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_sttmodel_transcribe_blocking: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_ttsmodel_synthesize()
		})
		if checksum != 59860 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_ttsmodel_synthesize: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_ttsmodel_synthesize_blocking()
		})
		if checksum != 50217 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_ttsmodel_synthesize_blocking: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_completionmodel_complete()
		})
		if checksum != 52439 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_completionmodel_complete: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_completionmodel_complete_blocking()
		})
		if checksum != 23180 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_completionmodel_complete_blocking: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_completionmodel_model_id()
		})
		if checksum != 47193 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_completionmodel_model_id: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_embeddingmodel_dimensions()
		})
		if checksum != 11198 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_embeddingmodel_dimensions: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_embeddingmodel_embed()
		})
		if checksum != 24214 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_embeddingmodel_embed: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_embeddingmodel_embed_blocking()
		})
		if checksum != 65533 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_embeddingmodel_embed_blocking: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_embeddingmodel_model_id()
		})
		if checksum != 36076 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_embeddingmodel_model_id: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_peerclient_node_id()
		})
		if checksum != 16043 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_peerclient_node_id: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_peerclient_run_remote_workflow()
		})
		if checksum != 48825 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_peerclient_run_remote_workflow: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_peerclient_run_remote_workflow_blocking()
		})
		if checksum != 52844 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_peerclient_run_remote_workflow_blocking: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_peerserver_serve()
		})
		if checksum != 3626 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_peerserver_serve: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_peerserver_serve_blocking()
		})
		if checksum != 61811 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_peerserver_serve_blocking: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_checkpointstore_delete()
		})
		if checksum != 19821 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_checkpointstore_delete: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_checkpointstore_delete_blocking()
		})
		if checksum != 18516 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_checkpointstore_delete_blocking: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_checkpointstore_list()
		})
		if checksum != 32714 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_checkpointstore_list: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_checkpointstore_list_blocking()
		})
		if checksum != 8449 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_checkpointstore_list_blocking: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_checkpointstore_list_run_ids()
		})
		if checksum != 47548 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_checkpointstore_list_run_ids: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_checkpointstore_list_run_ids_blocking()
		})
		if checksum != 19490 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_checkpointstore_list_run_ids_blocking: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_checkpointstore_load()
		})
		if checksum != 25287 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_checkpointstore_load: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_checkpointstore_load_blocking()
		})
		if checksum != 60522 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_checkpointstore_load_blocking: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_checkpointstore_save()
		})
		if checksum != 60168 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_checkpointstore_save: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_checkpointstore_save_blocking()
		})
		if checksum != 15819 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_checkpointstore_save_blocking: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_pipeline_run()
		})
		if checksum != 35674 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_pipeline_run: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_pipeline_run_blocking()
		})
		if checksum != 43680 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_pipeline_run_blocking: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_pipeline_stage_names()
		})
		if checksum != 24762 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_pipeline_stage_names: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_pipelinebuilder_add_workflow()
		})
		if checksum != 14561 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_pipelinebuilder_add_workflow: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_pipelinebuilder_build()
		})
		if checksum != 29058 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_pipelinebuilder_build: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_pipelinebuilder_parallel()
		})
		if checksum != 50044 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_pipelinebuilder_parallel: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_pipelinebuilder_stage()
		})
		if checksum != 785 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_pipelinebuilder_stage: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_pipelinebuilder_timeout_per_stage_ms()
		})
		if checksum != 11657 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_pipelinebuilder_timeout_per_stage_ms: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_pipelinebuilder_total_timeout_ms()
		})
		if checksum != 53032 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_pipelinebuilder_total_timeout_ms: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_baseprovider_as_completion_model()
		})
		if checksum != 16167 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_baseprovider_as_completion_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_baseprovider_defaults()
		})
		if checksum != 34325 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_baseprovider_defaults: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_baseprovider_extract()
		})
		if checksum != 47282 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_baseprovider_extract: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_baseprovider_model_id()
		})
		if checksum != 18871 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_baseprovider_model_id: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_baseprovider_with_defaults()
		})
		if checksum != 59299 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_baseprovider_with_defaults: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_baseprovider_with_response_format_json()
		})
		if checksum != 25582 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_baseprovider_with_response_format_json: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_baseprovider_with_system_prompt()
		})
		if checksum != 27915 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_baseprovider_with_system_prompt: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_baseprovider_with_tools_json()
		})
		if checksum != 43254 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_baseprovider_with_tools_json: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_customprovider_provider_id()
		})
		if checksum != 63939 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_customprovider_provider_id: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_customprovider_complete()
		})
		if checksum != 50396 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_customprovider_complete: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_customprovider_stream()
		})
		if checksum != 43804 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_customprovider_stream: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_customprovider_embed()
		})
		if checksum != 11956 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_customprovider_embed: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_customprovider_text_to_speech()
		})
		if checksum != 5652 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_customprovider_text_to_speech: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_customprovider_generate_music()
		})
		if checksum != 31843 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_customprovider_generate_music: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_customprovider_generate_sfx()
		})
		if checksum != 4679 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_customprovider_generate_sfx: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_customprovider_clone_voice()
		})
		if checksum != 49721 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_customprovider_clone_voice: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_customprovider_list_voices()
		})
		if checksum != 52071 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_customprovider_list_voices: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_customprovider_delete_voice()
		})
		if checksum != 32490 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_customprovider_delete_voice: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_customprovider_generate_image()
		})
		if checksum != 12529 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_customprovider_generate_image: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_customprovider_upscale_image()
		})
		if checksum != 11452 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_customprovider_upscale_image: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_customprovider_text_to_video()
		})
		if checksum != 49611 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_customprovider_text_to_video: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_customprovider_image_to_video()
		})
		if checksum != 28560 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_customprovider_image_to_video: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_customprovider_transcribe()
		})
		if checksum != 8428 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_customprovider_transcribe: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_customprovider_generate_3d()
		})
		if checksum != 49932 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_customprovider_generate_3d: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_customprovider_remove_background()
		})
		if checksum != 5224 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_customprovider_remove_background: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_customproviderhandle_as_base()
		})
		if checksum != 14798 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_customproviderhandle_as_base: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_customproviderhandle_clone_voice()
		})
		if checksum != 62032 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_customproviderhandle_clone_voice: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_customproviderhandle_complete()
		})
		if checksum != 8359 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_customproviderhandle_complete: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_customproviderhandle_delete_voice()
		})
		if checksum != 40671 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_customproviderhandle_delete_voice: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_customproviderhandle_embed()
		})
		if checksum != 25884 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_customproviderhandle_embed: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_customproviderhandle_generate_3d()
		})
		if checksum != 60831 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_customproviderhandle_generate_3d: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_customproviderhandle_generate_image()
		})
		if checksum != 4562 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_customproviderhandle_generate_image: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_customproviderhandle_generate_music()
		})
		if checksum != 30761 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_customproviderhandle_generate_music: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_customproviderhandle_generate_sfx()
		})
		if checksum != 16555 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_customproviderhandle_generate_sfx: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_customproviderhandle_image_to_video()
		})
		if checksum != 1304 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_customproviderhandle_image_to_video: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_customproviderhandle_list_voices()
		})
		if checksum != 20628 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_customproviderhandle_list_voices: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_customproviderhandle_provider_id()
		})
		if checksum != 2474 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_customproviderhandle_provider_id: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_customproviderhandle_remove_background()
		})
		if checksum != 30856 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_customproviderhandle_remove_background: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_customproviderhandle_stream()
		})
		if checksum != 19661 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_customproviderhandle_stream: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_customproviderhandle_text_to_speech()
		})
		if checksum != 20321 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_customproviderhandle_text_to_speech: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_customproviderhandle_text_to_video()
		})
		if checksum != 45453 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_customproviderhandle_text_to_video: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_customproviderhandle_transcribe()
		})
		if checksum != 57418 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_customproviderhandle_transcribe: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_customproviderhandle_upscale_image()
		})
		if checksum != 48926 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_customproviderhandle_upscale_image: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_customproviderhandle_with_response_format_json()
		})
		if checksum != 13552 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_customproviderhandle_with_response_format_json: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_customproviderhandle_with_system_prompt()
		})
		if checksum != 22875 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_customproviderhandle_with_system_prompt: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_customproviderhandle_with_tools_json()
		})
		if checksum != 18982 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_customproviderhandle_with_tools_json: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_completionstreamsink_on_chunk()
		})
		if checksum != 55655 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_completionstreamsink_on_chunk: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_completionstreamsink_on_done()
		})
		if checksum != 40753 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_completionstreamsink_on_done: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_completionstreamsink_on_error()
		})
		if checksum != 6341 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_completionstreamsink_on_error: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_stephandler_invoke()
		})
		if checksum != 11814 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_stephandler_invoke: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_workflow_run()
		})
		if checksum != 12733 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_workflow_run: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_workflow_run_blocking()
		})
		if checksum != 64927 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_workflow_run_blocking: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_workflow_step_names()
		})
		if checksum != 3893 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_workflow_step_names: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_workflowbuilder_build()
		})
		if checksum != 37017 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_workflowbuilder_build: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_workflowbuilder_step()
		})
		if checksum != 26605 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_workflowbuilder_step: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_workflowbuilder_step_timeout_ms()
		})
		if checksum != 8215 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_workflowbuilder_step_timeout_ms: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_workflowbuilder_timeout_ms()
		})
		if checksum != 61492 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_workflowbuilder_timeout_ms: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_constructor_agent_new()
		})
		if checksum != 30307 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_constructor_agent_new: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_constructor_peerclient_connect()
		})
		if checksum != 36996 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_constructor_peerclient_connect: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_constructor_peerserver_new()
		})
		if checksum != 57865 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_constructor_peerserver_new: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_constructor_pipelinebuilder_new()
		})
		if checksum != 61410 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_constructor_pipelinebuilder_new: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_constructor_baseprovider_from_completion_model()
		})
		if checksum != 9239 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_constructor_baseprovider_from_completion_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_constructor_baseprovider_with_completion_defaults()
		})
		if checksum != 23775 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_constructor_baseprovider_with_completion_defaults: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_constructor_workflowbuilder_new()
		})
		if checksum != 14241 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_constructor_workflowbuilder_new: UniFFI API checksum mismatch")
		}
	}
}

type FfiConverterUint16 struct{}

var FfiConverterUint16INSTANCE = FfiConverterUint16{}

func (FfiConverterUint16) Lower(value uint16) C.uint16_t {
	return C.uint16_t(value)
}

func (FfiConverterUint16) Write(writer io.Writer, value uint16) {
	writeUint16(writer, value)
}

func (FfiConverterUint16) Lift(value C.uint16_t) uint16 {
	return uint16(value)
}

func (FfiConverterUint16) Read(reader io.Reader) uint16 {
	return readUint16(reader)
}

type FfiDestroyerUint16 struct{}

func (FfiDestroyerUint16) Destroy(_ uint16) {}

type FfiConverterUint32 struct{}

var FfiConverterUint32INSTANCE = FfiConverterUint32{}

func (FfiConverterUint32) Lower(value uint32) C.uint32_t {
	return C.uint32_t(value)
}

func (FfiConverterUint32) Write(writer io.Writer, value uint32) {
	writeUint32(writer, value)
}

func (FfiConverterUint32) Lift(value C.uint32_t) uint32 {
	return uint32(value)
}

func (FfiConverterUint32) Read(reader io.Reader) uint32 {
	return readUint32(reader)
}

type FfiDestroyerUint32 struct{}

func (FfiDestroyerUint32) Destroy(_ uint32) {}

type FfiConverterUint64 struct{}

var FfiConverterUint64INSTANCE = FfiConverterUint64{}

func (FfiConverterUint64) Lower(value uint64) C.uint64_t {
	return C.uint64_t(value)
}

func (FfiConverterUint64) Write(writer io.Writer, value uint64) {
	writeUint64(writer, value)
}

func (FfiConverterUint64) Lift(value C.uint64_t) uint64 {
	return uint64(value)
}

func (FfiConverterUint64) Read(reader io.Reader) uint64 {
	return readUint64(reader)
}

type FfiDestroyerUint64 struct{}

func (FfiDestroyerUint64) Destroy(_ uint64) {}

type FfiConverterFloat32 struct{}

var FfiConverterFloat32INSTANCE = FfiConverterFloat32{}

func (FfiConverterFloat32) Lower(value float32) C.float {
	return C.float(value)
}

func (FfiConverterFloat32) Write(writer io.Writer, value float32) {
	writeFloat32(writer, value)
}

func (FfiConverterFloat32) Lift(value C.float) float32 {
	return float32(value)
}

func (FfiConverterFloat32) Read(reader io.Reader) float32 {
	return readFloat32(reader)
}

type FfiDestroyerFloat32 struct{}

func (FfiDestroyerFloat32) Destroy(_ float32) {}

type FfiConverterFloat64 struct{}

var FfiConverterFloat64INSTANCE = FfiConverterFloat64{}

func (FfiConverterFloat64) Lower(value float64) C.double {
	return C.double(value)
}

func (FfiConverterFloat64) Write(writer io.Writer, value float64) {
	writeFloat64(writer, value)
}

func (FfiConverterFloat64) Lift(value C.double) float64 {
	return float64(value)
}

func (FfiConverterFloat64) Read(reader io.Reader) float64 {
	return readFloat64(reader)
}

type FfiDestroyerFloat64 struct{}

func (FfiDestroyerFloat64) Destroy(_ float64) {}

type FfiConverterBool struct{}

var FfiConverterBoolINSTANCE = FfiConverterBool{}

func (FfiConverterBool) Lower(value bool) C.int8_t {
	if value {
		return C.int8_t(1)
	}
	return C.int8_t(0)
}

func (FfiConverterBool) Write(writer io.Writer, value bool) {
	if value {
		writeInt8(writer, 1)
	} else {
		writeInt8(writer, 0)
	}
}

func (FfiConverterBool) Lift(value C.int8_t) bool {
	return value != 0
}

func (FfiConverterBool) Read(reader io.Reader) bool {
	return readInt8(reader) != 0
}

type FfiDestroyerBool struct{}

func (FfiDestroyerBool) Destroy(_ bool) {}

type FfiConverterString struct{}

var FfiConverterStringINSTANCE = FfiConverterString{}

func (FfiConverterString) Lift(rb RustBufferI) string {
	defer rb.Free()
	reader := rb.AsReader()
	b, err := io.ReadAll(reader)
	if err != nil {
		panic(fmt.Errorf("reading reader: %w", err))
	}
	return string(b)
}

func (FfiConverterString) Read(reader io.Reader) string {
	length := readInt32(reader)
	buffer := make([]byte, length)
	read_length, err := reader.Read(buffer)
	if err != nil && err != io.EOF {
		panic(err)
	}
	if read_length != int(length) {
		panic(fmt.Errorf("bad read length when reading string, expected %d, read %d", length, read_length))
	}
	return string(buffer)
}

func (FfiConverterString) Lower(value string) C.RustBuffer {
	return stringToRustBuffer(value)
}

func (c FfiConverterString) LowerExternal(value string) ExternalCRustBuffer {
	return RustBufferFromC(stringToRustBuffer(value))
}

func (FfiConverterString) Write(writer io.Writer, value string) {
	if len(value) > math.MaxInt32 {
		panic("String is too large to fit into Int32")
	}

	writeInt32(writer, int32(len(value)))
	write_length, err := io.WriteString(writer, value)
	if err != nil {
		panic(err)
	}
	if write_length != len(value) {
		panic(fmt.Errorf("bad write length when writing string, expected %d, written %d", len(value), write_length))
	}
}

type FfiDestroyerString struct{}

func (FfiDestroyerString) Destroy(_ string) {}

// Below is an implementation of synchronization requirements outlined in the link.
// https://github.com/mozilla/uniffi-rs/blob/0dc031132d9493ca812c3af6e7dd60ad2ea95bf0/uniffi_bindgen/src/bindings/kotlin/templates/ObjectRuntime.kt#L31

type FfiObject struct {
	handle        C.uint64_t
	callCounter   atomic.Int64
	cloneFunction func(C.uint64_t, *C.RustCallStatus) C.uint64_t
	freeFunction  func(C.uint64_t, *C.RustCallStatus)
	destroyed     atomic.Bool
}

func newFfiObject(
	handle C.uint64_t,
	cloneFunction func(C.uint64_t, *C.RustCallStatus) C.uint64_t,
	freeFunction func(C.uint64_t, *C.RustCallStatus),
) FfiObject {
	return FfiObject{
		handle:        handle,
		cloneFunction: cloneFunction,
		freeFunction:  freeFunction,
	}
}

func (ffiObject *FfiObject) incrementPointer(debugName string) C.uint64_t {
	for {
		counter := ffiObject.callCounter.Load()
		if counter <= -1 {
			panic(fmt.Errorf("%v object has already been destroyed", debugName))
		}
		if counter == math.MaxInt64 {
			panic(fmt.Errorf("%v object call counter would overflow", debugName))
		}
		if ffiObject.callCounter.CompareAndSwap(counter, counter+1) {
			break
		}
	}

	return rustCall(func(status *C.RustCallStatus) C.uint64_t {
		return ffiObject.cloneFunction(ffiObject.handle, status)
	})
}

func (ffiObject *FfiObject) decrementPointer() {
	if ffiObject.callCounter.Add(-1) == -1 {
		ffiObject.freeRustArcPtr()
	}
}

func (ffiObject *FfiObject) destroy() {
	if ffiObject.destroyed.CompareAndSwap(false, true) {
		if ffiObject.callCounter.Add(-1) == -1 {
			ffiObject.freeRustArcPtr()
		}
	}
}

func (ffiObject *FfiObject) freeRustArcPtr() {
	if ffiObject.handle == 0 {
		return
	}
	rustCall(func(status *C.RustCallStatus) int32 {
		ffiObject.freeFunction(ffiObject.handle, status)
		return 0
	})
}

// A configured LLM agent that drives the standard tool-execution loop.
//
// Construct via [`Agent::new`] with a model, optional system prompt, the
// list of [`Tool`] definitions the model may call, a foreign-language
// [`ToolHandler`] that executes those tools, and a `max_iterations` budget.
// Then invoke [`run`](Self::run) (async) or
// [`run_blocking`](Self::run_blocking) (sync).
//
// Reuse a single `Agent` across calls when configuration is stable — the
// underlying model handle is reference-counted, so cloning is cheap.
type AgentInterface interface {
	// Run the agent loop until the model produces a final answer (no tool
	// calls) or `max_iterations` is reached.
	//
	// `user_input` is sent as the initial user-role message. The final
	// answer is returned in [`AgentResult::final_message`].
	Run(userInput string) (AgentResult, error)
	// Synchronous variant of [`run`](Self::run) — blocks the current thread
	// on the shared Tokio runtime.
	RunBlocking(userInput string) (AgentResult, error)
}

// A configured LLM agent that drives the standard tool-execution loop.
//
// Construct via [`Agent::new`] with a model, optional system prompt, the
// list of [`Tool`] definitions the model may call, a foreign-language
// [`ToolHandler`] that executes those tools, and a `max_iterations` budget.
// Then invoke [`run`](Self::run) (async) or
// [`run_blocking`](Self::run_blocking) (sync).
//
// Reuse a single `Agent` across calls when configuration is stable — the
// underlying model handle is reference-counted, so cloning is cheap.
type Agent struct {
	ffiObject FfiObject
}

// Build an agent.
//
// - `model`: the completion model to drive.
// - `system_prompt`: optional system prompt prepended to the conversation
// on every iteration.
// - `tools`: the tools the model may invoke. The names embedded in each
// [`Tool`] must match the names the [`ToolHandler`] dispatches on.
// - `tool_handler`: the foreign-language executor for tool calls.
// - `max_iterations`: hard cap on LLM round-trips before the loop is
// forced to produce a final answer.
func NewAgent(model *CompletionModel, systemPrompt *string, tools []Tool, toolHandler ToolHandler, maxIterations uint32) *Agent {
	return FfiConverterAgentINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_constructor_agent_new(FfiConverterCompletionModelINSTANCE.Lower(model), FfiConverterOptionalStringINSTANCE.Lower(systemPrompt), FfiConverterSequenceToolINSTANCE.Lower(tools), FfiConverterToolHandlerINSTANCE.Lower(toolHandler), FfiConverterUint32INSTANCE.Lower(maxIterations), _uniffiStatus)
	}))
}

// Run the agent loop until the model produces a final answer (no tool
// calls) or `max_iterations` is reached.
//
// `user_input` is sent as the initial user-role message. The final
// answer is returned in [`AgentResult::final_message`].
func (_self *Agent) Run(userInput string) (AgentResult, error) {
	_pointer := _self.ffiObject.incrementPointer("*Agent")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) AgentResult {
			return FfiConverterAgentResultINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_agent_run(
			_pointer, FfiConverterStringINSTANCE.Lower(userInput)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// Synchronous variant of [`run`](Self::run) — blocks the current thread
// on the shared Tokio runtime.
func (_self *Agent) RunBlocking(userInput string) (AgentResult, error) {
	_pointer := _self.ffiObject.incrementPointer("*Agent")
	defer _self.ffiObject.decrementPointer()
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) RustBufferI {
		return GoRustBuffer{
			inner: C.uniffi_blazen_uniffi_fn_method_agent_run_blocking(
				_pointer, FfiConverterStringINSTANCE.Lower(userInput), _uniffiStatus),
		}
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue AgentResult
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterAgentResultINSTANCE.Lift(_uniffiRV), nil
	}
}
func (object *Agent) Destroy() {
	runtime.SetFinalizer(object, nil)
	object.ffiObject.destroy()
}

type FfiConverterAgent struct{}

var FfiConverterAgentINSTANCE = FfiConverterAgent{}

func (c FfiConverterAgent) Lift(handle C.uint64_t) *Agent {
	result := &Agent{
		newFfiObject(
			handle,
			func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
				return C.uniffi_blazen_uniffi_fn_clone_agent(handle, status)
			},
			func(handle C.uint64_t, status *C.RustCallStatus) {
				C.uniffi_blazen_uniffi_fn_free_agent(handle, status)
			},
		),
	}
	runtime.SetFinalizer(result, (*Agent).Destroy)
	return result
}

func (c FfiConverterAgent) Read(reader io.Reader) *Agent {
	return c.Lift(C.uint64_t(readUint64(reader)))
}

func (c FfiConverterAgent) Lower(value *Agent) C.uint64_t {
	// TODO: this is bad - all synchronization from ObjectRuntime.go is discarded here,
	// because the handle will be decremented immediately after this function returns,
	// and someone will be left holding onto a non-locked handle.
	handle := value.ffiObject.incrementPointer("*Agent")
	defer value.ffiObject.decrementPointer()
	return handle
}

func (c FfiConverterAgent) Write(writer io.Writer, value *Agent) {
	writeUint64(writer, uint64(c.Lower(value)))
}

func LiftFromExternalAgent(handle uint64) *Agent {
	return FfiConverterAgentINSTANCE.Lift(C.uint64_t(handle))
}

func LowerToExternalAgent(value *Agent) uint64 {
	return uint64(FfiConverterAgentINSTANCE.Lower(value))
}

type FfiDestroyerAgent struct{}

func (_ FfiDestroyerAgent) Destroy(value *Agent) {
	value.Destroy()
}

// A [`crate::llm::CompletionModel`] wrapped with applied
// [`CompletionProviderDefaults`].
//
// Construct via [`BaseProvider::from_completion_model`] (wraps an existing
// model with no defaults) or [`BaseProvider::with_completion_defaults`]
// (wraps with explicit defaults). Mutate via the `with_*` builder methods.
//
// Phase B's `CustomProvider` factories will return `Arc<BaseProvider>`
// directly; for Phase A this class is reachable by lifting any existing
// `CompletionModel` factory result.
type BaseProviderInterface interface {
	// Unwrap to a plain [`CompletionModel`] handle that applies the
	// configured defaults on every call.
	//
	// Use this when you want to pass the wrapped provider to an API that
	// takes a generic `CompletionModel` (the agent runner, workflow
	// steps, etc.).
	AsCompletionModel() *CompletionModel
	// Inspect the currently-configured defaults (data only — hooks are
	// not surfaced in Phase A).
	Defaults() CompletionProviderDefaults
	// Extract structured output from the model by constraining its
	// response to a JSON Schema.
	//
	// Mirrors the upstream
	// [`blazen_llm::traits::StructuredOutput::extract`] blanket impl: the
	// `schema_json` is injected as the request's `response_format` and
	// the completion is dispatched as usual. Returns the model's raw
	// content (which the foreign caller deserializes into its own typed
	// shape — UniFFI cannot return a generic typed value across the FFI).
	//
	// `schema_json` must be a valid JSON Schema string; an empty string or
	// malformed JSON falls back to `null` (the request is sent without a
	// `response_format`).
	Extract(schemaJson string, messages []ChatMessage) (string, error)
	// The model id of the wrapped inner `CompletionModel`.
	ModelId() string
	// Replace the entire [`CompletionProviderDefaults`] on this provider,
	// returning a new `Arc<BaseProvider>` (clone-with-mutation).
	WithDefaults(defaults CompletionProviderDefaults) *BaseProvider
	// Set the default `response_format` (JSON-encoded `serde_json::Value`).
	//
	// Malformed JSON or an empty string is treated as JSON null.
	WithResponseFormatJson(fmtJson string) *BaseProvider
	// Set the default system prompt.
	WithSystemPrompt(prompt string) *BaseProvider
	// Set the default tools (JSON-encoded `Vec<ToolDefinition>`).
	//
	// Malformed JSON is treated as an empty tool list — matching the
	// upstream `#[derive(Default)]` semantics. Foreign callers should
	// validate the JSON before sending it across the FFI.
	WithToolsJson(toolsJson string) *BaseProvider
}

// A [`crate::llm::CompletionModel`] wrapped with applied
// [`CompletionProviderDefaults`].
//
// Construct via [`BaseProvider::from_completion_model`] (wraps an existing
// model with no defaults) or [`BaseProvider::with_completion_defaults`]
// (wraps with explicit defaults). Mutate via the `with_*` builder methods.
//
// Phase B's `CustomProvider` factories will return `Arc<BaseProvider>`
// directly; for Phase A this class is reachable by lifting any existing
// `CompletionModel` factory result.
type BaseProvider struct {
	ffiObject FfiObject
}

// Wrap an existing [`CompletionModel`] with empty defaults.
//
// Equivalent to using the wrapped model directly, but lets callers
// attach defaults later via the `with_*` methods.
func BaseProviderFromCompletionModel(model *CompletionModel) *BaseProvider {
	return FfiConverterBaseProviderINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_constructor_baseprovider_from_completion_model(FfiConverterCompletionModelINSTANCE.Lower(model), _uniffiStatus)
	}))
}

// Wrap a [`CompletionModel`] with explicit
// [`CompletionProviderDefaults`].
func BaseProviderWithCompletionDefaults(model *CompletionModel, defaults CompletionProviderDefaults) *BaseProvider {
	return FfiConverterBaseProviderINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_constructor_baseprovider_with_completion_defaults(FfiConverterCompletionModelINSTANCE.Lower(model), FfiConverterCompletionProviderDefaultsINSTANCE.Lower(defaults), _uniffiStatus)
	}))
}

// Unwrap to a plain [`CompletionModel`] handle that applies the
// configured defaults on every call.
//
// Use this when you want to pass the wrapped provider to an API that
// takes a generic `CompletionModel` (the agent runner, workflow
// steps, etc.).
func (_self *BaseProvider) AsCompletionModel() *CompletionModel {
	_pointer := _self.ffiObject.incrementPointer("*BaseProvider")
	defer _self.ffiObject.decrementPointer()
	return FfiConverterCompletionModelINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_method_baseprovider_as_completion_model(
			_pointer, _uniffiStatus)
	}))
}

// Inspect the currently-configured defaults (data only — hooks are
// not surfaced in Phase A).
func (_self *BaseProvider) Defaults() CompletionProviderDefaults {
	_pointer := _self.ffiObject.incrementPointer("*BaseProvider")
	defer _self.ffiObject.decrementPointer()
	return FfiConverterCompletionProviderDefaultsINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) RustBufferI {
		return GoRustBuffer{
			inner: C.uniffi_blazen_uniffi_fn_method_baseprovider_defaults(
				_pointer, _uniffiStatus),
		}
	}))
}

// Extract structured output from the model by constraining its
// response to a JSON Schema.
//
// Mirrors the upstream
// [`blazen_llm::traits::StructuredOutput::extract`] blanket impl: the
// `schema_json` is injected as the request's `response_format` and
// the completion is dispatched as usual. Returns the model's raw
// content (which the foreign caller deserializes into its own typed
// shape — UniFFI cannot return a generic typed value across the FFI).
//
// `schema_json` must be a valid JSON Schema string; an empty string or
// malformed JSON falls back to `null` (the request is sent without a
// `response_format`).
func (_self *BaseProvider) Extract(schemaJson string, messages []ChatMessage) (string, error) {
	_pointer := _self.ffiObject.incrementPointer("*BaseProvider")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) string {
			return FfiConverterStringINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_baseprovider_extract(
			_pointer, FfiConverterStringINSTANCE.Lower(schemaJson), FfiConverterSequenceChatMessageINSTANCE.Lower(messages)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// The model id of the wrapped inner `CompletionModel`.
func (_self *BaseProvider) ModelId() string {
	_pointer := _self.ffiObject.incrementPointer("*BaseProvider")
	defer _self.ffiObject.decrementPointer()
	return FfiConverterStringINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) RustBufferI {
		return GoRustBuffer{
			inner: C.uniffi_blazen_uniffi_fn_method_baseprovider_model_id(
				_pointer, _uniffiStatus),
		}
	}))
}

// Replace the entire [`CompletionProviderDefaults`] on this provider,
// returning a new `Arc<BaseProvider>` (clone-with-mutation).
func (_self *BaseProvider) WithDefaults(defaults CompletionProviderDefaults) *BaseProvider {
	_pointer := _self.ffiObject.incrementPointer("*BaseProvider")
	defer _self.ffiObject.decrementPointer()
	return FfiConverterBaseProviderINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_method_baseprovider_with_defaults(
			_pointer, FfiConverterCompletionProviderDefaultsINSTANCE.Lower(defaults), _uniffiStatus)
	}))
}

// Set the default `response_format` (JSON-encoded `serde_json::Value`).
//
// Malformed JSON or an empty string is treated as JSON null.
func (_self *BaseProvider) WithResponseFormatJson(fmtJson string) *BaseProvider {
	_pointer := _self.ffiObject.incrementPointer("*BaseProvider")
	defer _self.ffiObject.decrementPointer()
	return FfiConverterBaseProviderINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_method_baseprovider_with_response_format_json(
			_pointer, FfiConverterStringINSTANCE.Lower(fmtJson), _uniffiStatus)
	}))
}

// Set the default system prompt.
func (_self *BaseProvider) WithSystemPrompt(prompt string) *BaseProvider {
	_pointer := _self.ffiObject.incrementPointer("*BaseProvider")
	defer _self.ffiObject.decrementPointer()
	return FfiConverterBaseProviderINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_method_baseprovider_with_system_prompt(
			_pointer, FfiConverterStringINSTANCE.Lower(prompt), _uniffiStatus)
	}))
}

// Set the default tools (JSON-encoded `Vec<ToolDefinition>`).
//
// Malformed JSON is treated as an empty tool list — matching the
// upstream `#[derive(Default)]` semantics. Foreign callers should
// validate the JSON before sending it across the FFI.
func (_self *BaseProvider) WithToolsJson(toolsJson string) *BaseProvider {
	_pointer := _self.ffiObject.incrementPointer("*BaseProvider")
	defer _self.ffiObject.decrementPointer()
	return FfiConverterBaseProviderINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_method_baseprovider_with_tools_json(
			_pointer, FfiConverterStringINSTANCE.Lower(toolsJson), _uniffiStatus)
	}))
}
func (object *BaseProvider) Destroy() {
	runtime.SetFinalizer(object, nil)
	object.ffiObject.destroy()
}

type FfiConverterBaseProvider struct{}

var FfiConverterBaseProviderINSTANCE = FfiConverterBaseProvider{}

func (c FfiConverterBaseProvider) Lift(handle C.uint64_t) *BaseProvider {
	result := &BaseProvider{
		newFfiObject(
			handle,
			func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
				return C.uniffi_blazen_uniffi_fn_clone_baseprovider(handle, status)
			},
			func(handle C.uint64_t, status *C.RustCallStatus) {
				C.uniffi_blazen_uniffi_fn_free_baseprovider(handle, status)
			},
		),
	}
	runtime.SetFinalizer(result, (*BaseProvider).Destroy)
	return result
}

func (c FfiConverterBaseProvider) Read(reader io.Reader) *BaseProvider {
	return c.Lift(C.uint64_t(readUint64(reader)))
}

func (c FfiConverterBaseProvider) Lower(value *BaseProvider) C.uint64_t {
	// TODO: this is bad - all synchronization from ObjectRuntime.go is discarded here,
	// because the handle will be decremented immediately after this function returns,
	// and someone will be left holding onto a non-locked handle.
	handle := value.ffiObject.incrementPointer("*BaseProvider")
	defer value.ffiObject.decrementPointer()
	return handle
}

func (c FfiConverterBaseProvider) Write(writer io.Writer, value *BaseProvider) {
	writeUint64(writer, uint64(c.Lower(value)))
}

func LiftFromExternalBaseProvider(handle uint64) *BaseProvider {
	return FfiConverterBaseProviderINSTANCE.Lift(C.uint64_t(handle))
}

func LowerToExternalBaseProvider(value *BaseProvider) uint64 {
	return uint64(FfiConverterBaseProviderINSTANCE.Lower(value))
}

type FfiDestroyerBaseProvider struct{}

func (_ FfiDestroyerBaseProvider) Destroy(value *BaseProvider) {
	value.Destroy()
}

// A workflow-checkpoint store handle.
//
// Wraps any [`blazen_persist::CheckpointStore`] implementation behind a
// uniform FFI surface. Construct via:
//
// - [`new_redb_checkpoint_store`] for an embedded file-backed store.
// - [`new_valkey_checkpoint_store`] for a Redis/ValKey server-backed store.
//
// Each method has both an async variant (recommended on Swift / Kotlin /
// modern Ruby fibers) and a `_blocking` variant (handy for Go `main`
// functions and quick scripts).
type CheckpointStoreInterface interface {
	// Delete the checkpoint for the given run id (UUID string). Succeeds
	// even when no checkpoint exists for the id (the underlying backends
	// treat delete-of-missing as a no-op).
	Delete(runId string) error
	// Synchronous variant of [`delete`](Self::delete).
	DeleteBlocking(runId string) error
	// List all stored checkpoints, ordered by timestamp descending (most
	// recent first).
	List() ([]WorkflowCheckpoint, error)
	// Synchronous variant of [`list`](Self::list).
	ListBlocking() ([]WorkflowCheckpoint, error)
	// List all stored run ids (as UUID strings), ordered by timestamp
	// descending (most recent first).
	//
	// Cheaper than [`list`](Self::list) when callers only need to
	// enumerate ids — but note that the underlying backend still loads
	// each checkpoint to read its timestamp for ordering, so this is a
	// convenience wrapper rather than a true index scan.
	ListRunIds() ([]string, error)
	// Synchronous variant of [`list_run_ids`](Self::list_run_ids).
	ListRunIdsBlocking() ([]string, error)
	// Load a checkpoint by its run id (UUID string). Returns `None` when no
	// checkpoint exists for the given id.
	Load(runId string) (*WorkflowCheckpoint, error)
	// Synchronous variant of [`load`](Self::load).
	LoadBlocking(runId string) (*WorkflowCheckpoint, error)
	// Persist a checkpoint, overwriting any existing entry with the same
	// `run_id`. Async on Swift / Kotlin; blocking-with-suspension on Go.
	Save(checkpoint WorkflowCheckpoint) error
	// Synchronous variant of [`save`](Self::save).
	SaveBlocking(checkpoint WorkflowCheckpoint) error
}

// A workflow-checkpoint store handle.
//
// Wraps any [`blazen_persist::CheckpointStore`] implementation behind a
// uniform FFI surface. Construct via:
//
// - [`new_redb_checkpoint_store`] for an embedded file-backed store.
// - [`new_valkey_checkpoint_store`] for a Redis/ValKey server-backed store.
//
// Each method has both an async variant (recommended on Swift / Kotlin /
// modern Ruby fibers) and a `_blocking` variant (handy for Go `main`
// functions and quick scripts).
type CheckpointStore struct {
	ffiObject FfiObject
}

// Delete the checkpoint for the given run id (UUID string). Succeeds
// even when no checkpoint exists for the id (the underlying backends
// treat delete-of-missing as a no-op).
func (_self *CheckpointStore) Delete(runId string) error {
	_pointer := _self.ffiObject.incrementPointer("*CheckpointStore")
	defer _self.ffiObject.decrementPointer()
	_, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) struct{} {
			C.ffi_blazen_uniffi_rust_future_complete_void(handle, status)
			return struct{}{}
		},
		// liftFn
		func(_ struct{}) struct{} { return struct{}{} },
		C.uniffi_blazen_uniffi_fn_method_checkpointstore_delete(
			_pointer, FfiConverterStringINSTANCE.Lower(runId)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_void(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_void(handle)
		},
	)

	if err == nil {
		return nil
	}

	return err
}

// Synchronous variant of [`delete`](Self::delete).
func (_self *CheckpointStore) DeleteBlocking(runId string) error {
	_pointer := _self.ffiObject.incrementPointer("*CheckpointStore")
	defer _self.ffiObject.decrementPointer()
	_, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) bool {
		C.uniffi_blazen_uniffi_fn_method_checkpointstore_delete_blocking(
			_pointer, FfiConverterStringINSTANCE.Lower(runId), _uniffiStatus)
		return false
	})
	return _uniffiErr.AsError()
}

// List all stored checkpoints, ordered by timestamp descending (most
// recent first).
func (_self *CheckpointStore) List() ([]WorkflowCheckpoint, error) {
	_pointer := _self.ffiObject.incrementPointer("*CheckpointStore")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) []WorkflowCheckpoint {
			return FfiConverterSequenceWorkflowCheckpointINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_checkpointstore_list(
			_pointer),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// Synchronous variant of [`list`](Self::list).
func (_self *CheckpointStore) ListBlocking() ([]WorkflowCheckpoint, error) {
	_pointer := _self.ffiObject.incrementPointer("*CheckpointStore")
	defer _self.ffiObject.decrementPointer()
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) RustBufferI {
		return GoRustBuffer{
			inner: C.uniffi_blazen_uniffi_fn_method_checkpointstore_list_blocking(
				_pointer, _uniffiStatus),
		}
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue []WorkflowCheckpoint
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterSequenceWorkflowCheckpointINSTANCE.Lift(_uniffiRV), nil
	}
}

// List all stored run ids (as UUID strings), ordered by timestamp
// descending (most recent first).
//
// Cheaper than [`list`](Self::list) when callers only need to
// enumerate ids — but note that the underlying backend still loads
// each checkpoint to read its timestamp for ordering, so this is a
// convenience wrapper rather than a true index scan.
func (_self *CheckpointStore) ListRunIds() ([]string, error) {
	_pointer := _self.ffiObject.incrementPointer("*CheckpointStore")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) []string {
			return FfiConverterSequenceStringINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_checkpointstore_list_run_ids(
			_pointer),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// Synchronous variant of [`list_run_ids`](Self::list_run_ids).
func (_self *CheckpointStore) ListRunIdsBlocking() ([]string, error) {
	_pointer := _self.ffiObject.incrementPointer("*CheckpointStore")
	defer _self.ffiObject.decrementPointer()
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) RustBufferI {
		return GoRustBuffer{
			inner: C.uniffi_blazen_uniffi_fn_method_checkpointstore_list_run_ids_blocking(
				_pointer, _uniffiStatus),
		}
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue []string
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterSequenceStringINSTANCE.Lift(_uniffiRV), nil
	}
}

// Load a checkpoint by its run id (UUID string). Returns `None` when no
// checkpoint exists for the given id.
func (_self *CheckpointStore) Load(runId string) (*WorkflowCheckpoint, error) {
	_pointer := _self.ffiObject.incrementPointer("*CheckpointStore")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) *WorkflowCheckpoint {
			return FfiConverterOptionalWorkflowCheckpointINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_checkpointstore_load(
			_pointer, FfiConverterStringINSTANCE.Lower(runId)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// Synchronous variant of [`load`](Self::load).
func (_self *CheckpointStore) LoadBlocking(runId string) (*WorkflowCheckpoint, error) {
	_pointer := _self.ffiObject.incrementPointer("*CheckpointStore")
	defer _self.ffiObject.decrementPointer()
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) RustBufferI {
		return GoRustBuffer{
			inner: C.uniffi_blazen_uniffi_fn_method_checkpointstore_load_blocking(
				_pointer, FfiConverterStringINSTANCE.Lower(runId), _uniffiStatus),
		}
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *WorkflowCheckpoint
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterOptionalWorkflowCheckpointINSTANCE.Lift(_uniffiRV), nil
	}
}

// Persist a checkpoint, overwriting any existing entry with the same
// `run_id`. Async on Swift / Kotlin; blocking-with-suspension on Go.
func (_self *CheckpointStore) Save(checkpoint WorkflowCheckpoint) error {
	_pointer := _self.ffiObject.incrementPointer("*CheckpointStore")
	defer _self.ffiObject.decrementPointer()
	_, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) struct{} {
			C.ffi_blazen_uniffi_rust_future_complete_void(handle, status)
			return struct{}{}
		},
		// liftFn
		func(_ struct{}) struct{} { return struct{}{} },
		C.uniffi_blazen_uniffi_fn_method_checkpointstore_save(
			_pointer, FfiConverterWorkflowCheckpointINSTANCE.Lower(checkpoint)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_void(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_void(handle)
		},
	)

	if err == nil {
		return nil
	}

	return err
}

// Synchronous variant of [`save`](Self::save).
func (_self *CheckpointStore) SaveBlocking(checkpoint WorkflowCheckpoint) error {
	_pointer := _self.ffiObject.incrementPointer("*CheckpointStore")
	defer _self.ffiObject.decrementPointer()
	_, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) bool {
		C.uniffi_blazen_uniffi_fn_method_checkpointstore_save_blocking(
			_pointer, FfiConverterWorkflowCheckpointINSTANCE.Lower(checkpoint), _uniffiStatus)
		return false
	})
	return _uniffiErr.AsError()
}
func (object *CheckpointStore) Destroy() {
	runtime.SetFinalizer(object, nil)
	object.ffiObject.destroy()
}

type FfiConverterCheckpointStore struct{}

var FfiConverterCheckpointStoreINSTANCE = FfiConverterCheckpointStore{}

func (c FfiConverterCheckpointStore) Lift(handle C.uint64_t) *CheckpointStore {
	result := &CheckpointStore{
		newFfiObject(
			handle,
			func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
				return C.uniffi_blazen_uniffi_fn_clone_checkpointstore(handle, status)
			},
			func(handle C.uint64_t, status *C.RustCallStatus) {
				C.uniffi_blazen_uniffi_fn_free_checkpointstore(handle, status)
			},
		),
	}
	runtime.SetFinalizer(result, (*CheckpointStore).Destroy)
	return result
}

func (c FfiConverterCheckpointStore) Read(reader io.Reader) *CheckpointStore {
	return c.Lift(C.uint64_t(readUint64(reader)))
}

func (c FfiConverterCheckpointStore) Lower(value *CheckpointStore) C.uint64_t {
	// TODO: this is bad - all synchronization from ObjectRuntime.go is discarded here,
	// because the handle will be decremented immediately after this function returns,
	// and someone will be left holding onto a non-locked handle.
	handle := value.ffiObject.incrementPointer("*CheckpointStore")
	defer value.ffiObject.decrementPointer()
	return handle
}

func (c FfiConverterCheckpointStore) Write(writer io.Writer, value *CheckpointStore) {
	writeUint64(writer, uint64(c.Lower(value)))
}

func LiftFromExternalCheckpointStore(handle uint64) *CheckpointStore {
	return FfiConverterCheckpointStoreINSTANCE.Lift(C.uint64_t(handle))
}

func LowerToExternalCheckpointStore(value *CheckpointStore) uint64 {
	return uint64(FfiConverterCheckpointStoreINSTANCE.Lower(value))
}

type FfiDestroyerCheckpointStore struct{}

func (_ FfiDestroyerCheckpointStore) Destroy(value *CheckpointStore) {
	value.Destroy()
}

// A chat completion model.
//
// Construct one via the per-provider factories in `providers.rs` (e.g.
// `CompletionModel::openai(options)` from the foreign-language side).
// Once obtained, call [`complete`](Self::complete) (async) or
// [`complete_blocking`](Self::complete_blocking) (sync) to generate
// responses.
type CompletionModelInterface interface {
	// Perform a chat completion. Async on Swift / Kotlin; blocking on Go
	// (UniFFI's Go bindgen wraps the future in a goroutine-friendly call).
	Complete(request CompletionRequest) (CompletionResponse, error)
	// Synchronous variant of [`complete`](Self::complete) — blocks the
	// current thread on the shared Tokio runtime. Handy for Ruby scripts
	// and quick Go `main` functions where async machinery is overkill.
	// Prefer the async [`complete`](Self::complete) in long-running services.
	CompleteBlocking(request CompletionRequest) (CompletionResponse, error)
	// The model's identifier (e.g. `"gpt-4o"`, `"claude-3-5-sonnet"`).
	ModelId() string
}

// A chat completion model.
//
// Construct one via the per-provider factories in `providers.rs` (e.g.
// `CompletionModel::openai(options)` from the foreign-language side).
// Once obtained, call [`complete`](Self::complete) (async) or
// [`complete_blocking`](Self::complete_blocking) (sync) to generate
// responses.
type CompletionModel struct {
	ffiObject FfiObject
}

// Perform a chat completion. Async on Swift / Kotlin; blocking on Go
// (UniFFI's Go bindgen wraps the future in a goroutine-friendly call).
func (_self *CompletionModel) Complete(request CompletionRequest) (CompletionResponse, error) {
	_pointer := _self.ffiObject.incrementPointer("*CompletionModel")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) CompletionResponse {
			return FfiConverterCompletionResponseINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_completionmodel_complete(
			_pointer, FfiConverterCompletionRequestINSTANCE.Lower(request)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// Synchronous variant of [`complete`](Self::complete) — blocks the
// current thread on the shared Tokio runtime. Handy for Ruby scripts
// and quick Go `main` functions where async machinery is overkill.
// Prefer the async [`complete`](Self::complete) in long-running services.
func (_self *CompletionModel) CompleteBlocking(request CompletionRequest) (CompletionResponse, error) {
	_pointer := _self.ffiObject.incrementPointer("*CompletionModel")
	defer _self.ffiObject.decrementPointer()
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) RustBufferI {
		return GoRustBuffer{
			inner: C.uniffi_blazen_uniffi_fn_method_completionmodel_complete_blocking(
				_pointer, FfiConverterCompletionRequestINSTANCE.Lower(request), _uniffiStatus),
		}
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue CompletionResponse
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterCompletionResponseINSTANCE.Lift(_uniffiRV), nil
	}
}

// The model's identifier (e.g. `"gpt-4o"`, `"claude-3-5-sonnet"`).
func (_self *CompletionModel) ModelId() string {
	_pointer := _self.ffiObject.incrementPointer("*CompletionModel")
	defer _self.ffiObject.decrementPointer()
	return FfiConverterStringINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) RustBufferI {
		return GoRustBuffer{
			inner: C.uniffi_blazen_uniffi_fn_method_completionmodel_model_id(
				_pointer, _uniffiStatus),
		}
	}))
}
func (object *CompletionModel) Destroy() {
	runtime.SetFinalizer(object, nil)
	object.ffiObject.destroy()
}

type FfiConverterCompletionModel struct{}

var FfiConverterCompletionModelINSTANCE = FfiConverterCompletionModel{}

func (c FfiConverterCompletionModel) Lift(handle C.uint64_t) *CompletionModel {
	result := &CompletionModel{
		newFfiObject(
			handle,
			func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
				return C.uniffi_blazen_uniffi_fn_clone_completionmodel(handle, status)
			},
			func(handle C.uint64_t, status *C.RustCallStatus) {
				C.uniffi_blazen_uniffi_fn_free_completionmodel(handle, status)
			},
		),
	}
	runtime.SetFinalizer(result, (*CompletionModel).Destroy)
	return result
}

func (c FfiConverterCompletionModel) Read(reader io.Reader) *CompletionModel {
	return c.Lift(C.uint64_t(readUint64(reader)))
}

func (c FfiConverterCompletionModel) Lower(value *CompletionModel) C.uint64_t {
	// TODO: this is bad - all synchronization from ObjectRuntime.go is discarded here,
	// because the handle will be decremented immediately after this function returns,
	// and someone will be left holding onto a non-locked handle.
	handle := value.ffiObject.incrementPointer("*CompletionModel")
	defer value.ffiObject.decrementPointer()
	return handle
}

func (c FfiConverterCompletionModel) Write(writer io.Writer, value *CompletionModel) {
	writeUint64(writer, uint64(c.Lower(value)))
}

func LiftFromExternalCompletionModel(handle uint64) *CompletionModel {
	return FfiConverterCompletionModelINSTANCE.Lift(C.uint64_t(handle))
}

func LowerToExternalCompletionModel(value *CompletionModel) uint64 {
	return uint64(FfiConverterCompletionModelINSTANCE.Lower(value))
}

type FfiDestroyerCompletionModel struct{}

func (_ FfiDestroyerCompletionModel) Destroy(value *CompletionModel) {
	value.Destroy()
}

// Sink for streaming chat completion output, implemented in foreign code.
//
// The streaming engine calls [`on_chunk`](Self::on_chunk) for each chunk as
// it arrives, then exactly one of [`on_done`](Self::on_done) (success) or
// [`on_error`](Self::on_error) (failure). Implementations should treat
// `on_done` / `on_error` as cleanup hooks (close channels, complete async
// iterators, etc.).
//
// ## Async story
//
// All three methods are `async` on the Rust side. UniFFI exposes them as:
// - Go: blocking functions, safe from goroutines (compose with channels)
// - Swift: `async throws` methods
// - Kotlin: `suspend fun` methods
// - Ruby: blocking methods (wrap in `Async { ... }` for fiber concurrency)
type CompletionStreamSink interface {
	// Receive a single chunk from the streaming response.
	//
	// Returning an `Err` aborts the stream — the streaming engine will not
	// call further `on_chunk` callbacks and will not call `on_done`.
	OnChunk(chunk StreamChunk) error
	// Receive the terminal completion signal. Called exactly once at the
	// end of a successful stream. Implementations should perform any
	// cleanup here (close channels, signal completion to async iterators).
	//
	// `finish_reason` is the last `finish_reason` reported by the
	// provider (e.g. `"stop"`, `"tool_calls"`, `"length"`) — empty string
	// when the provider didn't report one. `usage` is the running token
	// usage; some providers don't surface usage via the stream, in which
	// case all counters are zero.
	OnDone(finishReason string, usage TokenUsage) error
	// Receive a fatal error from the stream. Called exactly once when the
	// stream fails midway. After `on_error` fires, neither further
	// `on_chunk` nor `on_done` will be called.
	OnError(err *BlazenError) error
}

// Sink for streaming chat completion output, implemented in foreign code.
//
// The streaming engine calls [`on_chunk`](Self::on_chunk) for each chunk as
// it arrives, then exactly one of [`on_done`](Self::on_done) (success) or
// [`on_error`](Self::on_error) (failure). Implementations should treat
// `on_done` / `on_error` as cleanup hooks (close channels, complete async
// iterators, etc.).
//
// ## Async story
//
// All three methods are `async` on the Rust side. UniFFI exposes them as:
// - Go: blocking functions, safe from goroutines (compose with channels)
// - Swift: `async throws` methods
// - Kotlin: `suspend fun` methods
// - Ruby: blocking methods (wrap in `Async { ... }` for fiber concurrency)
type CompletionStreamSinkImpl struct {
	ffiObject FfiObject
}

// Receive a single chunk from the streaming response.
//
// Returning an `Err` aborts the stream — the streaming engine will not
// call further `on_chunk` callbacks and will not call `on_done`.
func (_self *CompletionStreamSinkImpl) OnChunk(chunk StreamChunk) error {
	_pointer := _self.ffiObject.incrementPointer("CompletionStreamSink")
	defer _self.ffiObject.decrementPointer()
	_, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) struct{} {
			C.ffi_blazen_uniffi_rust_future_complete_void(handle, status)
			return struct{}{}
		},
		// liftFn
		func(_ struct{}) struct{} { return struct{}{} },
		C.uniffi_blazen_uniffi_fn_method_completionstreamsink_on_chunk(
			_pointer, FfiConverterStreamChunkINSTANCE.Lower(chunk)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_void(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_void(handle)
		},
	)

	if err == nil {
		return nil
	}

	return err
}

// Receive the terminal completion signal. Called exactly once at the
// end of a successful stream. Implementations should perform any
// cleanup here (close channels, signal completion to async iterators).
//
// `finish_reason` is the last `finish_reason` reported by the
// provider (e.g. `"stop"`, `"tool_calls"`, `"length"`) — empty string
// when the provider didn't report one. `usage` is the running token
// usage; some providers don't surface usage via the stream, in which
// case all counters are zero.
func (_self *CompletionStreamSinkImpl) OnDone(finishReason string, usage TokenUsage) error {
	_pointer := _self.ffiObject.incrementPointer("CompletionStreamSink")
	defer _self.ffiObject.decrementPointer()
	_, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) struct{} {
			C.ffi_blazen_uniffi_rust_future_complete_void(handle, status)
			return struct{}{}
		},
		// liftFn
		func(_ struct{}) struct{} { return struct{}{} },
		C.uniffi_blazen_uniffi_fn_method_completionstreamsink_on_done(
			_pointer, FfiConverterStringINSTANCE.Lower(finishReason), FfiConverterTokenUsageINSTANCE.Lower(usage)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_void(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_void(handle)
		},
	)

	if err == nil {
		return nil
	}

	return err
}

// Receive a fatal error from the stream. Called exactly once when the
// stream fails midway. After `on_error` fires, neither further
// `on_chunk` nor `on_done` will be called.
func (_self *CompletionStreamSinkImpl) OnError(onErrorArg *BlazenError) error {
	_pointer := _self.ffiObject.incrementPointer("CompletionStreamSink")
	defer _self.ffiObject.decrementPointer()
	_, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) struct{} {
			C.ffi_blazen_uniffi_rust_future_complete_void(handle, status)
			return struct{}{}
		},
		// liftFn
		func(_ struct{}) struct{} { return struct{}{} },
		C.uniffi_blazen_uniffi_fn_method_completionstreamsink_on_error(
			_pointer, FfiConverterBlazenErrorINSTANCE.Lower(onErrorArg)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_void(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_void(handle)
		},
	)

	if err == nil {
		return nil
	}

	return err
}
func (object *CompletionStreamSinkImpl) Destroy() {
	runtime.SetFinalizer(object, nil)
	object.ffiObject.destroy()
}

type FfiConverterCompletionStreamSink struct {
	handleMap *concurrentHandleMap[CompletionStreamSink]
}

var FfiConverterCompletionStreamSinkINSTANCE = FfiConverterCompletionStreamSink{
	handleMap: newConcurrentHandleMap[CompletionStreamSink](),
}

func (c FfiConverterCompletionStreamSink) Lift(handle C.uint64_t) CompletionStreamSink {
	if uint64(handle)&1 == 0 {
		// Rust-generated handle (even), construct a new object wrapping the handle
		result := &CompletionStreamSinkImpl{
			newFfiObject(
				handle,
				func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
					return C.uniffi_blazen_uniffi_fn_clone_completionstreamsink(handle, status)
				},
				func(handle C.uint64_t, status *C.RustCallStatus) {
					C.uniffi_blazen_uniffi_fn_free_completionstreamsink(handle, status)
				},
			),
		}
		runtime.SetFinalizer(result, (*CompletionStreamSinkImpl).Destroy)
		return result
	} else {
		// Go-generated handle (odd), retrieve from the handle map
		val, ok := c.handleMap.tryGet(uint64(handle))
		if !ok {
			panic(fmt.Errorf("no callback in handle map: %d", handle))
		}
		c.handleMap.remove(uint64(handle))
		return val
	}
}

func (c FfiConverterCompletionStreamSink) Read(reader io.Reader) CompletionStreamSink {
	return c.Lift(C.uint64_t(readUint64(reader)))
}

func (c FfiConverterCompletionStreamSink) Lower(value CompletionStreamSink) C.uint64_t {
	// TODO: this is bad - all synchronization from ObjectRuntime.go is discarded here,
	// because the handle will be decremented immediately after this function returns,
	// and someone will be left holding onto a non-locked handle.
	if val, ok := value.(*CompletionStreamSinkImpl); ok {
		// Rust-backed object, clone the handle
		handle := val.ffiObject.incrementPointer("CompletionStreamSink")
		defer val.ffiObject.decrementPointer()
		return handle
	} else {
		// Go-backed object, insert into handle map
		return C.uint64_t(c.handleMap.insert(value))
	}
}

func (c FfiConverterCompletionStreamSink) Write(writer io.Writer, value CompletionStreamSink) {
	writeUint64(writer, uint64(c.Lower(value)))
}

func LiftFromExternalCompletionStreamSink(handle uint64) CompletionStreamSink {
	return FfiConverterCompletionStreamSinkINSTANCE.Lift(C.uint64_t(handle))
}

func LowerToExternalCompletionStreamSink(value CompletionStreamSink) uint64 {
	return uint64(FfiConverterCompletionStreamSinkINSTANCE.Lower(value))
}

type FfiDestroyerCompletionStreamSink struct{}

func (_ FfiDestroyerCompletionStreamSink) Destroy(value CompletionStreamSink) {
	if val, ok := value.(*CompletionStreamSinkImpl); ok {
		val.Destroy()
	}
}

type uniffiCallbackResult C.int8_t

const (
	uniffiIdxCallbackFree               uniffiCallbackResult = 0
	uniffiCallbackResultSuccess         uniffiCallbackResult = 0
	uniffiCallbackResultError           uniffiCallbackResult = 1
	uniffiCallbackUnexpectedResultError uniffiCallbackResult = 2
	uniffiCallbackCancelled             uniffiCallbackResult = 3
)

type concurrentHandleMap[T any] struct {
	handles       map[uint64]T
	currentHandle uint64
	lock          sync.RWMutex
}

func newConcurrentHandleMap[T any]() *concurrentHandleMap[T] {
	return &concurrentHandleMap[T]{
		handles:       map[uint64]T{},
		currentHandle: 1,
	}
}

func (cm *concurrentHandleMap[T]) insert(obj T) uint64 {
	cm.lock.Lock()
	defer cm.lock.Unlock()

	handle := cm.currentHandle
	cm.currentHandle = cm.currentHandle + 2
	cm.handles[handle] = obj
	return handle
}

func (cm *concurrentHandleMap[T]) remove(handle uint64) {
	cm.lock.Lock()
	defer cm.lock.Unlock()

	delete(cm.handles, handle)
}

func (cm *concurrentHandleMap[T]) tryGet(handle uint64) (T, bool) {
	cm.lock.RLock()
	defer cm.lock.RUnlock()

	val, ok := cm.handles[handle]
	return val, ok
}

//export blazen_uniffi_streaming_cgo_dispatchCallbackInterfaceCompletionStreamSinkMethod0
func blazen_uniffi_streaming_cgo_dispatchCallbackInterfaceCompletionStreamSinkMethod0(uniffiHandle C.uint64_t, chunk C.RustBuffer, uniffiFutureCallback C.UniffiForeignFutureCompleteVoid, uniffiCallbackData C.uint64_t, uniffiOutDroppedCallback *C.UniffiForeignFutureDroppedCallbackStruct) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterCompletionStreamSinkINSTANCE.handleMap.tryGet(handle)
	if !ok {
		panic(fmt.Errorf("no callback in handle map: %d", handle))
	}

	result := make(chan C.UniffiForeignFutureResultVoid, 1)
	cancel := make(chan struct{}, 1)
	guardHandle := cgo.NewHandle(cancel)
	*uniffiOutDroppedCallback = C.UniffiForeignFutureDroppedCallbackStruct{
		handle: C.uint64_t(guardHandle),
		free:   C.UniffiForeignFutureDroppedCallback(C.blazen_uniffiFreeGorutine),
	}

	// Wait for compleation or cancel
	go func() {
		select {
		case <-cancel:
		case res := <-result:
			C.call_UniffiForeignFutureCompleteVoid(uniffiFutureCallback, uniffiCallbackData, res)
		}
	}()

	// Eval callback asynchroniously
	go func() {
		asyncResult := &C.UniffiForeignFutureResultVoid{}
		callStatus := &asyncResult.callStatus
		defer func() {
			result <- *asyncResult
		}()

		err :=
			uniffiObj.OnChunk(
				FfiConverterStreamChunkINSTANCE.Lift(GoRustBuffer{
					inner: chunk,
				}),
			)

		if err != nil {
			var actualError *BlazenError
			if errors.As(err, &actualError) {
				*callStatus = C.RustCallStatus{
					code:     C.int8_t(uniffiCallbackResultError),
					errorBuf: FfiConverterBlazenErrorINSTANCE.Lower(actualError),
				}
			} else {
				*callStatus = C.RustCallStatus{
					code: C.int8_t(uniffiCallbackUnexpectedResultError),
				}
			}
			return
		}

	}()
}

//export blazen_uniffi_streaming_cgo_dispatchCallbackInterfaceCompletionStreamSinkMethod1
func blazen_uniffi_streaming_cgo_dispatchCallbackInterfaceCompletionStreamSinkMethod1(uniffiHandle C.uint64_t, finishReason C.RustBuffer, usage C.RustBuffer, uniffiFutureCallback C.UniffiForeignFutureCompleteVoid, uniffiCallbackData C.uint64_t, uniffiOutDroppedCallback *C.UniffiForeignFutureDroppedCallbackStruct) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterCompletionStreamSinkINSTANCE.handleMap.tryGet(handle)
	if !ok {
		panic(fmt.Errorf("no callback in handle map: %d", handle))
	}

	result := make(chan C.UniffiForeignFutureResultVoid, 1)
	cancel := make(chan struct{}, 1)
	guardHandle := cgo.NewHandle(cancel)
	*uniffiOutDroppedCallback = C.UniffiForeignFutureDroppedCallbackStruct{
		handle: C.uint64_t(guardHandle),
		free:   C.UniffiForeignFutureDroppedCallback(C.blazen_uniffiFreeGorutine),
	}

	// Wait for compleation or cancel
	go func() {
		select {
		case <-cancel:
		case res := <-result:
			C.call_UniffiForeignFutureCompleteVoid(uniffiFutureCallback, uniffiCallbackData, res)
		}
	}()

	// Eval callback asynchroniously
	go func() {
		asyncResult := &C.UniffiForeignFutureResultVoid{}
		callStatus := &asyncResult.callStatus
		defer func() {
			result <- *asyncResult
		}()

		err :=
			uniffiObj.OnDone(
				FfiConverterStringINSTANCE.Lift(GoRustBuffer{
					inner: finishReason,
				}),
				FfiConverterTokenUsageINSTANCE.Lift(GoRustBuffer{
					inner: usage,
				}),
			)

		if err != nil {
			var actualError *BlazenError
			if errors.As(err, &actualError) {
				*callStatus = C.RustCallStatus{
					code:     C.int8_t(uniffiCallbackResultError),
					errorBuf: FfiConverterBlazenErrorINSTANCE.Lower(actualError),
				}
			} else {
				*callStatus = C.RustCallStatus{
					code: C.int8_t(uniffiCallbackUnexpectedResultError),
				}
			}
			return
		}

	}()
}

//export blazen_uniffi_streaming_cgo_dispatchCallbackInterfaceCompletionStreamSinkMethod2
func blazen_uniffi_streaming_cgo_dispatchCallbackInterfaceCompletionStreamSinkMethod2(uniffiHandle C.uint64_t, err C.RustBuffer, uniffiFutureCallback C.UniffiForeignFutureCompleteVoid, uniffiCallbackData C.uint64_t, uniffiOutDroppedCallback *C.UniffiForeignFutureDroppedCallbackStruct) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterCompletionStreamSinkINSTANCE.handleMap.tryGet(handle)
	if !ok {
		panic(fmt.Errorf("no callback in handle map: %d", handle))
	}

	result := make(chan C.UniffiForeignFutureResultVoid, 1)
	cancel := make(chan struct{}, 1)
	guardHandle := cgo.NewHandle(cancel)
	*uniffiOutDroppedCallback = C.UniffiForeignFutureDroppedCallbackStruct{
		handle: C.uint64_t(guardHandle),
		free:   C.UniffiForeignFutureDroppedCallback(C.blazen_uniffiFreeGorutine),
	}

	// Wait for compleation or cancel
	go func() {
		select {
		case <-cancel:
		case res := <-result:
			C.call_UniffiForeignFutureCompleteVoid(uniffiFutureCallback, uniffiCallbackData, res)
		}
	}()

	// Eval callback asynchroniously
	go func() {
		asyncResult := &C.UniffiForeignFutureResultVoid{}
		callStatus := &asyncResult.callStatus
		defer func() {
			result <- *asyncResult
		}()

		err :=
			uniffiObj.OnError(
				FfiConverterBlazenErrorINSTANCE.Lift(GoRustBuffer{
					inner: err,
				}),
			)

		if err != nil {
			var actualError *BlazenError
			if errors.As(err, &actualError) {
				*callStatus = C.RustCallStatus{
					code:     C.int8_t(uniffiCallbackResultError),
					errorBuf: FfiConverterBlazenErrorINSTANCE.Lower(actualError),
				}
			} else {
				*callStatus = C.RustCallStatus{
					code: C.int8_t(uniffiCallbackUnexpectedResultError),
				}
			}
			return
		}

	}()
}

var UniffiVTableCallbackInterfaceCompletionStreamSinkINSTANCE = C.UniffiVTableCallbackInterfaceCompletionStreamSink{
	uniffiFree:  (C.UniffiCallbackInterfaceFree)(C.blazen_uniffi_streaming_cgo_dispatchCallbackInterfaceCompletionStreamSinkFree),
	uniffiClone: (C.UniffiCallbackInterfaceClone)(C.blazen_uniffi_streaming_cgo_dispatchCallbackInterfaceCompletionStreamSinkClone),
	onChunk:     (C.UniffiCallbackInterfaceCompletionStreamSinkMethod0)(C.blazen_uniffi_streaming_cgo_dispatchCallbackInterfaceCompletionStreamSinkMethod0),
	onDone:      (C.UniffiCallbackInterfaceCompletionStreamSinkMethod1)(C.blazen_uniffi_streaming_cgo_dispatchCallbackInterfaceCompletionStreamSinkMethod1),
	onError:     (C.UniffiCallbackInterfaceCompletionStreamSinkMethod2)(C.blazen_uniffi_streaming_cgo_dispatchCallbackInterfaceCompletionStreamSinkMethod2),
}

//export blazen_uniffi_streaming_cgo_dispatchCallbackInterfaceCompletionStreamSinkFree
func blazen_uniffi_streaming_cgo_dispatchCallbackInterfaceCompletionStreamSinkFree(handle C.uint64_t) {
	FfiConverterCompletionStreamSinkINSTANCE.handleMap.remove(uint64(handle))
}

//export blazen_uniffi_streaming_cgo_dispatchCallbackInterfaceCompletionStreamSinkClone
func blazen_uniffi_streaming_cgo_dispatchCallbackInterfaceCompletionStreamSinkClone(handle C.uint64_t) C.uint64_t {
	val, ok := FfiConverterCompletionStreamSinkINSTANCE.handleMap.tryGet(uint64(handle))
	if !ok {
		panic(fmt.Errorf("no callback in handle map: %d", handle))
	}
	return C.uint64_t(FfiConverterCompletionStreamSinkINSTANCE.handleMap.insert(val))
}

func (c FfiConverterCompletionStreamSink) register() {
	C.uniffi_blazen_uniffi_fn_init_callback_vtable_completionstreamsink(&UniffiVTableCallbackInterfaceCompletionStreamSinkINSTANCE)
}

// User-extensible provider trait the foreign side implements directly.
//
// Mirrors [`blazen_llm::CustomProvider`] across the UniFFI boundary. Has 16
// typed async methods (completion, streaming-via-sink, embeddings, plus 13
// compute / media methods) and one sync `provider_id` accessor.
//
// ## How foreign users use it
//
// Foreign users implement this trait on their own type and pass an instance
// to [`custom_provider_from_foreign`] to obtain a [`CustomProviderHandle`]
// usable wherever Blazen expects a provider.
//
// UniFFI's `with_foreign` traits require every method to be implemented at
// the foreign language level — there is no cross-FFI Rust "default impl"
// fallback. Each language binding ships a base class / extension that
// supplies throwing `Unsupported` defaults so users only need to override
// the capabilities their provider actually supports.
//
// ## Wire-format shape
//
// All argument and return types are UniFFI Records ([`SpeechRequest`],
// [`AudioResult`], ...) defined in [`crate::compute_types`]. The
// [`UniffiToCoreCustomProviderAdapter`] converts these to the upstream
// [`blazen_llm::compute`] types on each call.
//
// ## Async story
//
// Every method except [`provider_id`](Self::provider_id) is `async` on the
// Rust side. UniFFI exposes the methods as:
// - Go: blocking functions, safe from goroutines (compose with channels)
// - Swift: `async throws` methods
// - Kotlin: `suspend fun` methods
// - Ruby: blocking methods (wrap in `Async { ... }` block for fiber concurrency)
type CustomProvider interface {
	// Stable provider identifier for logs and metrics.
	ProviderId() string
	// Perform a non-streaming chat completion.
	Complete(request CompletionRequest) (CompletionResponse, error)
	// Perform a streaming chat completion, pushing chunks into the supplied
	// sink. The implementation must call `sink.on_done` exactly once on
	// success or `sink.on_error` exactly once on failure.
	Stream(request CompletionRequest, sink CompletionStreamSink) error
	// Embed one or more texts.
	Embed(texts []string) (EmbeddingResponse, error)
	// Synthesize speech from text.
	TextToSpeech(request SpeechRequest) (AudioResult, error)
	// Generate music from a prompt.
	GenerateMusic(request MusicRequest) (AudioResult, error)
	// Generate sound effects from a prompt.
	GenerateSfx(request MusicRequest) (AudioResult, error)
	// Clone a voice from reference audio.
	CloneVoice(request VoiceCloneRequest) (VoiceHandle, error)
	// List voices known to the provider.
	ListVoices() ([]VoiceHandle, error)
	// Delete a previously-cloned voice.
	DeleteVoice(voice VoiceHandle) error
	// Generate images from a prompt.
	GenerateImage(request ImageRequest) (ImageResult, error)
	// Upscale an existing image.
	UpscaleImage(request UpscaleRequest) (ImageResult, error)
	// Generate a video from a text prompt.
	TextToVideo(request VideoRequest) (VideoResult, error)
	// Generate a video from a source image + prompt.
	ImageToVideo(request VideoRequest) (VideoResult, error)
	// Transcribe audio to text.
	Transcribe(request TranscriptionRequest) (TranscriptionResult, error)
	// Generate a 3D model.
	Generate3d(request ThreeDRequest) (ThreeDResult, error)
	// Remove the background from an image.
	RemoveBackground(request BackgroundRemovalRequest) (ImageResult, error)
}

// User-extensible provider trait the foreign side implements directly.
//
// Mirrors [`blazen_llm::CustomProvider`] across the UniFFI boundary. Has 16
// typed async methods (completion, streaming-via-sink, embeddings, plus 13
// compute / media methods) and one sync `provider_id` accessor.
//
// ## How foreign users use it
//
// Foreign users implement this trait on their own type and pass an instance
// to [`custom_provider_from_foreign`] to obtain a [`CustomProviderHandle`]
// usable wherever Blazen expects a provider.
//
// UniFFI's `with_foreign` traits require every method to be implemented at
// the foreign language level — there is no cross-FFI Rust "default impl"
// fallback. Each language binding ships a base class / extension that
// supplies throwing `Unsupported` defaults so users only need to override
// the capabilities their provider actually supports.
//
// ## Wire-format shape
//
// All argument and return types are UniFFI Records ([`SpeechRequest`],
// [`AudioResult`], ...) defined in [`crate::compute_types`]. The
// [`UniffiToCoreCustomProviderAdapter`] converts these to the upstream
// [`blazen_llm::compute`] types on each call.
//
// ## Async story
//
// Every method except [`provider_id`](Self::provider_id) is `async` on the
// Rust side. UniFFI exposes the methods as:
// - Go: blocking functions, safe from goroutines (compose with channels)
// - Swift: `async throws` methods
// - Kotlin: `suspend fun` methods
// - Ruby: blocking methods (wrap in `Async { ... }` block for fiber concurrency)
type CustomProviderImpl struct {
	ffiObject FfiObject
}

// Stable provider identifier for logs and metrics.
func (_self *CustomProviderImpl) ProviderId() string {
	_pointer := _self.ffiObject.incrementPointer("CustomProvider")
	defer _self.ffiObject.decrementPointer()
	return FfiConverterStringINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) RustBufferI {
		return GoRustBuffer{
			inner: C.uniffi_blazen_uniffi_fn_method_customprovider_provider_id(
				_pointer, _uniffiStatus),
		}
	}))
}

// Perform a non-streaming chat completion.
func (_self *CustomProviderImpl) Complete(request CompletionRequest) (CompletionResponse, error) {
	_pointer := _self.ffiObject.incrementPointer("CustomProvider")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) CompletionResponse {
			return FfiConverterCompletionResponseINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_customprovider_complete(
			_pointer, FfiConverterCompletionRequestINSTANCE.Lower(request)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// Perform a streaming chat completion, pushing chunks into the supplied
// sink. The implementation must call `sink.on_done` exactly once on
// success or `sink.on_error` exactly once on failure.
func (_self *CustomProviderImpl) Stream(request CompletionRequest, sink CompletionStreamSink) error {
	_pointer := _self.ffiObject.incrementPointer("CustomProvider")
	defer _self.ffiObject.decrementPointer()
	_, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) struct{} {
			C.ffi_blazen_uniffi_rust_future_complete_void(handle, status)
			return struct{}{}
		},
		// liftFn
		func(_ struct{}) struct{} { return struct{}{} },
		C.uniffi_blazen_uniffi_fn_method_customprovider_stream(
			_pointer, FfiConverterCompletionRequestINSTANCE.Lower(request), FfiConverterCompletionStreamSinkINSTANCE.Lower(sink)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_void(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_void(handle)
		},
	)

	if err == nil {
		return nil
	}

	return err
}

// Embed one or more texts.
func (_self *CustomProviderImpl) Embed(texts []string) (EmbeddingResponse, error) {
	_pointer := _self.ffiObject.incrementPointer("CustomProvider")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) EmbeddingResponse {
			return FfiConverterEmbeddingResponseINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_customprovider_embed(
			_pointer, FfiConverterSequenceStringINSTANCE.Lower(texts)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// Synthesize speech from text.
func (_self *CustomProviderImpl) TextToSpeech(request SpeechRequest) (AudioResult, error) {
	_pointer := _self.ffiObject.incrementPointer("CustomProvider")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) AudioResult {
			return FfiConverterAudioResultINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_customprovider_text_to_speech(
			_pointer, FfiConverterSpeechRequestINSTANCE.Lower(request)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// Generate music from a prompt.
func (_self *CustomProviderImpl) GenerateMusic(request MusicRequest) (AudioResult, error) {
	_pointer := _self.ffiObject.incrementPointer("CustomProvider")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) AudioResult {
			return FfiConverterAudioResultINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_customprovider_generate_music(
			_pointer, FfiConverterMusicRequestINSTANCE.Lower(request)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// Generate sound effects from a prompt.
func (_self *CustomProviderImpl) GenerateSfx(request MusicRequest) (AudioResult, error) {
	_pointer := _self.ffiObject.incrementPointer("CustomProvider")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) AudioResult {
			return FfiConverterAudioResultINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_customprovider_generate_sfx(
			_pointer, FfiConverterMusicRequestINSTANCE.Lower(request)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// Clone a voice from reference audio.
func (_self *CustomProviderImpl) CloneVoice(request VoiceCloneRequest) (VoiceHandle, error) {
	_pointer := _self.ffiObject.incrementPointer("CustomProvider")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) VoiceHandle {
			return FfiConverterVoiceHandleINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_customprovider_clone_voice(
			_pointer, FfiConverterVoiceCloneRequestINSTANCE.Lower(request)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// List voices known to the provider.
func (_self *CustomProviderImpl) ListVoices() ([]VoiceHandle, error) {
	_pointer := _self.ffiObject.incrementPointer("CustomProvider")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) []VoiceHandle {
			return FfiConverterSequenceVoiceHandleINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_customprovider_list_voices(
			_pointer),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// Delete a previously-cloned voice.
func (_self *CustomProviderImpl) DeleteVoice(voice VoiceHandle) error {
	_pointer := _self.ffiObject.incrementPointer("CustomProvider")
	defer _self.ffiObject.decrementPointer()
	_, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) struct{} {
			C.ffi_blazen_uniffi_rust_future_complete_void(handle, status)
			return struct{}{}
		},
		// liftFn
		func(_ struct{}) struct{} { return struct{}{} },
		C.uniffi_blazen_uniffi_fn_method_customprovider_delete_voice(
			_pointer, FfiConverterVoiceHandleINSTANCE.Lower(voice)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_void(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_void(handle)
		},
	)

	if err == nil {
		return nil
	}

	return err
}

// Generate images from a prompt.
func (_self *CustomProviderImpl) GenerateImage(request ImageRequest) (ImageResult, error) {
	_pointer := _self.ffiObject.incrementPointer("CustomProvider")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) ImageResult {
			return FfiConverterImageResultINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_customprovider_generate_image(
			_pointer, FfiConverterImageRequestINSTANCE.Lower(request)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// Upscale an existing image.
func (_self *CustomProviderImpl) UpscaleImage(request UpscaleRequest) (ImageResult, error) {
	_pointer := _self.ffiObject.incrementPointer("CustomProvider")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) ImageResult {
			return FfiConverterImageResultINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_customprovider_upscale_image(
			_pointer, FfiConverterUpscaleRequestINSTANCE.Lower(request)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// Generate a video from a text prompt.
func (_self *CustomProviderImpl) TextToVideo(request VideoRequest) (VideoResult, error) {
	_pointer := _self.ffiObject.incrementPointer("CustomProvider")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) VideoResult {
			return FfiConverterVideoResultINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_customprovider_text_to_video(
			_pointer, FfiConverterVideoRequestINSTANCE.Lower(request)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// Generate a video from a source image + prompt.
func (_self *CustomProviderImpl) ImageToVideo(request VideoRequest) (VideoResult, error) {
	_pointer := _self.ffiObject.incrementPointer("CustomProvider")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) VideoResult {
			return FfiConverterVideoResultINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_customprovider_image_to_video(
			_pointer, FfiConverterVideoRequestINSTANCE.Lower(request)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// Transcribe audio to text.
func (_self *CustomProviderImpl) Transcribe(request TranscriptionRequest) (TranscriptionResult, error) {
	_pointer := _self.ffiObject.incrementPointer("CustomProvider")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) TranscriptionResult {
			return FfiConverterTranscriptionResultINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_customprovider_transcribe(
			_pointer, FfiConverterTranscriptionRequestINSTANCE.Lower(request)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// Generate a 3D model.
func (_self *CustomProviderImpl) Generate3d(request ThreeDRequest) (ThreeDResult, error) {
	_pointer := _self.ffiObject.incrementPointer("CustomProvider")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) ThreeDResult {
			return FfiConverterThreeDResultINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_customprovider_generate_3d(
			_pointer, FfiConverterThreeDRequestINSTANCE.Lower(request)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// Remove the background from an image.
func (_self *CustomProviderImpl) RemoveBackground(request BackgroundRemovalRequest) (ImageResult, error) {
	_pointer := _self.ffiObject.incrementPointer("CustomProvider")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) ImageResult {
			return FfiConverterImageResultINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_customprovider_remove_background(
			_pointer, FfiConverterBackgroundRemovalRequestINSTANCE.Lower(request)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}
func (object *CustomProviderImpl) Destroy() {
	runtime.SetFinalizer(object, nil)
	object.ffiObject.destroy()
}

type FfiConverterCustomProvider struct {
	handleMap *concurrentHandleMap[CustomProvider]
}

var FfiConverterCustomProviderINSTANCE = FfiConverterCustomProvider{
	handleMap: newConcurrentHandleMap[CustomProvider](),
}

func (c FfiConverterCustomProvider) Lift(handle C.uint64_t) CustomProvider {
	if uint64(handle)&1 == 0 {
		// Rust-generated handle (even), construct a new object wrapping the handle
		result := &CustomProviderImpl{
			newFfiObject(
				handle,
				func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
					return C.uniffi_blazen_uniffi_fn_clone_customprovider(handle, status)
				},
				func(handle C.uint64_t, status *C.RustCallStatus) {
					C.uniffi_blazen_uniffi_fn_free_customprovider(handle, status)
				},
			),
		}
		runtime.SetFinalizer(result, (*CustomProviderImpl).Destroy)
		return result
	} else {
		// Go-generated handle (odd), retrieve from the handle map
		val, ok := c.handleMap.tryGet(uint64(handle))
		if !ok {
			panic(fmt.Errorf("no callback in handle map: %d", handle))
		}
		c.handleMap.remove(uint64(handle))
		return val
	}
}

func (c FfiConverterCustomProvider) Read(reader io.Reader) CustomProvider {
	return c.Lift(C.uint64_t(readUint64(reader)))
}

func (c FfiConverterCustomProvider) Lower(value CustomProvider) C.uint64_t {
	// TODO: this is bad - all synchronization from ObjectRuntime.go is discarded here,
	// because the handle will be decremented immediately after this function returns,
	// and someone will be left holding onto a non-locked handle.
	if val, ok := value.(*CustomProviderImpl); ok {
		// Rust-backed object, clone the handle
		handle := val.ffiObject.incrementPointer("CustomProvider")
		defer val.ffiObject.decrementPointer()
		return handle
	} else {
		// Go-backed object, insert into handle map
		return C.uint64_t(c.handleMap.insert(value))
	}
}

func (c FfiConverterCustomProvider) Write(writer io.Writer, value CustomProvider) {
	writeUint64(writer, uint64(c.Lower(value)))
}

func LiftFromExternalCustomProvider(handle uint64) CustomProvider {
	return FfiConverterCustomProviderINSTANCE.Lift(C.uint64_t(handle))
}

func LowerToExternalCustomProvider(value CustomProvider) uint64 {
	return uint64(FfiConverterCustomProviderINSTANCE.Lower(value))
}

type FfiDestroyerCustomProvider struct{}

func (_ FfiDestroyerCustomProvider) Destroy(value CustomProvider) {
	if val, ok := value.(*CustomProviderImpl); ok {
		val.Destroy()
	}
}

//export blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod0
func blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod0(uniffiHandle C.uint64_t, uniffiOutReturn *C.RustBuffer, callStatus *C.RustCallStatus) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterCustomProviderINSTANCE.handleMap.tryGet(handle)
	if !ok {
		panic(fmt.Errorf("no callback in handle map: %d", handle))
	}

	res :=
		uniffiObj.ProviderId()

	*uniffiOutReturn = FfiConverterStringINSTANCE.Lower(res)
}

//export blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod1
func blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod1(uniffiHandle C.uint64_t, request C.RustBuffer, uniffiFutureCallback C.UniffiForeignFutureCompleteRustBuffer, uniffiCallbackData C.uint64_t, uniffiOutDroppedCallback *C.UniffiForeignFutureDroppedCallbackStruct) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterCustomProviderINSTANCE.handleMap.tryGet(handle)
	if !ok {
		panic(fmt.Errorf("no callback in handle map: %d", handle))
	}

	result := make(chan C.UniffiForeignFutureResultRustBuffer, 1)
	cancel := make(chan struct{}, 1)
	guardHandle := cgo.NewHandle(cancel)
	*uniffiOutDroppedCallback = C.UniffiForeignFutureDroppedCallbackStruct{
		handle: C.uint64_t(guardHandle),
		free:   C.UniffiForeignFutureDroppedCallback(C.blazen_uniffiFreeGorutine),
	}

	// Wait for compleation or cancel
	go func() {
		select {
		case <-cancel:
		case res := <-result:
			C.call_UniffiForeignFutureCompleteRustBuffer(uniffiFutureCallback, uniffiCallbackData, res)
		}
	}()

	// Eval callback asynchroniously
	go func() {
		asyncResult := &C.UniffiForeignFutureResultRustBuffer{}
		uniffiOutReturn := &asyncResult.returnValue
		callStatus := &asyncResult.callStatus
		defer func() {
			result <- *asyncResult
		}()

		res, err :=
			uniffiObj.Complete(
				FfiConverterCompletionRequestINSTANCE.Lift(GoRustBuffer{
					inner: request,
				}),
			)

		if err != nil {
			var actualError *BlazenError
			if errors.As(err, &actualError) {
				*callStatus = C.RustCallStatus{
					code:     C.int8_t(uniffiCallbackResultError),
					errorBuf: FfiConverterBlazenErrorINSTANCE.Lower(actualError),
				}
			} else {
				*callStatus = C.RustCallStatus{
					code: C.int8_t(uniffiCallbackUnexpectedResultError),
				}
			}
			return
		}

		*uniffiOutReturn = FfiConverterCompletionResponseINSTANCE.Lower(res)
	}()
}

//export blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod2
func blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod2(uniffiHandle C.uint64_t, request C.RustBuffer, sink C.uint64_t, uniffiFutureCallback C.UniffiForeignFutureCompleteVoid, uniffiCallbackData C.uint64_t, uniffiOutDroppedCallback *C.UniffiForeignFutureDroppedCallbackStruct) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterCustomProviderINSTANCE.handleMap.tryGet(handle)
	if !ok {
		panic(fmt.Errorf("no callback in handle map: %d", handle))
	}

	result := make(chan C.UniffiForeignFutureResultVoid, 1)
	cancel := make(chan struct{}, 1)
	guardHandle := cgo.NewHandle(cancel)
	*uniffiOutDroppedCallback = C.UniffiForeignFutureDroppedCallbackStruct{
		handle: C.uint64_t(guardHandle),
		free:   C.UniffiForeignFutureDroppedCallback(C.blazen_uniffiFreeGorutine),
	}

	// Wait for compleation or cancel
	go func() {
		select {
		case <-cancel:
		case res := <-result:
			C.call_UniffiForeignFutureCompleteVoid(uniffiFutureCallback, uniffiCallbackData, res)
		}
	}()

	// Eval callback asynchroniously
	go func() {
		asyncResult := &C.UniffiForeignFutureResultVoid{}
		callStatus := &asyncResult.callStatus
		defer func() {
			result <- *asyncResult
		}()

		err :=
			uniffiObj.Stream(
				FfiConverterCompletionRequestINSTANCE.Lift(GoRustBuffer{
					inner: request,
				}),
				FfiConverterCompletionStreamSinkINSTANCE.Lift(sink),
			)

		if err != nil {
			var actualError *BlazenError
			if errors.As(err, &actualError) {
				*callStatus = C.RustCallStatus{
					code:     C.int8_t(uniffiCallbackResultError),
					errorBuf: FfiConverterBlazenErrorINSTANCE.Lower(actualError),
				}
			} else {
				*callStatus = C.RustCallStatus{
					code: C.int8_t(uniffiCallbackUnexpectedResultError),
				}
			}
			return
		}

	}()
}

//export blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod3
func blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod3(uniffiHandle C.uint64_t, texts C.RustBuffer, uniffiFutureCallback C.UniffiForeignFutureCompleteRustBuffer, uniffiCallbackData C.uint64_t, uniffiOutDroppedCallback *C.UniffiForeignFutureDroppedCallbackStruct) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterCustomProviderINSTANCE.handleMap.tryGet(handle)
	if !ok {
		panic(fmt.Errorf("no callback in handle map: %d", handle))
	}

	result := make(chan C.UniffiForeignFutureResultRustBuffer, 1)
	cancel := make(chan struct{}, 1)
	guardHandle := cgo.NewHandle(cancel)
	*uniffiOutDroppedCallback = C.UniffiForeignFutureDroppedCallbackStruct{
		handle: C.uint64_t(guardHandle),
		free:   C.UniffiForeignFutureDroppedCallback(C.blazen_uniffiFreeGorutine),
	}

	// Wait for compleation or cancel
	go func() {
		select {
		case <-cancel:
		case res := <-result:
			C.call_UniffiForeignFutureCompleteRustBuffer(uniffiFutureCallback, uniffiCallbackData, res)
		}
	}()

	// Eval callback asynchroniously
	go func() {
		asyncResult := &C.UniffiForeignFutureResultRustBuffer{}
		uniffiOutReturn := &asyncResult.returnValue
		callStatus := &asyncResult.callStatus
		defer func() {
			result <- *asyncResult
		}()

		res, err :=
			uniffiObj.Embed(
				FfiConverterSequenceStringINSTANCE.Lift(GoRustBuffer{
					inner: texts,
				}),
			)

		if err != nil {
			var actualError *BlazenError
			if errors.As(err, &actualError) {
				*callStatus = C.RustCallStatus{
					code:     C.int8_t(uniffiCallbackResultError),
					errorBuf: FfiConverterBlazenErrorINSTANCE.Lower(actualError),
				}
			} else {
				*callStatus = C.RustCallStatus{
					code: C.int8_t(uniffiCallbackUnexpectedResultError),
				}
			}
			return
		}

		*uniffiOutReturn = FfiConverterEmbeddingResponseINSTANCE.Lower(res)
	}()
}

//export blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod4
func blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod4(uniffiHandle C.uint64_t, request C.RustBuffer, uniffiFutureCallback C.UniffiForeignFutureCompleteRustBuffer, uniffiCallbackData C.uint64_t, uniffiOutDroppedCallback *C.UniffiForeignFutureDroppedCallbackStruct) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterCustomProviderINSTANCE.handleMap.tryGet(handle)
	if !ok {
		panic(fmt.Errorf("no callback in handle map: %d", handle))
	}

	result := make(chan C.UniffiForeignFutureResultRustBuffer, 1)
	cancel := make(chan struct{}, 1)
	guardHandle := cgo.NewHandle(cancel)
	*uniffiOutDroppedCallback = C.UniffiForeignFutureDroppedCallbackStruct{
		handle: C.uint64_t(guardHandle),
		free:   C.UniffiForeignFutureDroppedCallback(C.blazen_uniffiFreeGorutine),
	}

	// Wait for compleation or cancel
	go func() {
		select {
		case <-cancel:
		case res := <-result:
			C.call_UniffiForeignFutureCompleteRustBuffer(uniffiFutureCallback, uniffiCallbackData, res)
		}
	}()

	// Eval callback asynchroniously
	go func() {
		asyncResult := &C.UniffiForeignFutureResultRustBuffer{}
		uniffiOutReturn := &asyncResult.returnValue
		callStatus := &asyncResult.callStatus
		defer func() {
			result <- *asyncResult
		}()

		res, err :=
			uniffiObj.TextToSpeech(
				FfiConverterSpeechRequestINSTANCE.Lift(GoRustBuffer{
					inner: request,
				}),
			)

		if err != nil {
			var actualError *BlazenError
			if errors.As(err, &actualError) {
				*callStatus = C.RustCallStatus{
					code:     C.int8_t(uniffiCallbackResultError),
					errorBuf: FfiConverterBlazenErrorINSTANCE.Lower(actualError),
				}
			} else {
				*callStatus = C.RustCallStatus{
					code: C.int8_t(uniffiCallbackUnexpectedResultError),
				}
			}
			return
		}

		*uniffiOutReturn = FfiConverterAudioResultINSTANCE.Lower(res)
	}()
}

//export blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod5
func blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod5(uniffiHandle C.uint64_t, request C.RustBuffer, uniffiFutureCallback C.UniffiForeignFutureCompleteRustBuffer, uniffiCallbackData C.uint64_t, uniffiOutDroppedCallback *C.UniffiForeignFutureDroppedCallbackStruct) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterCustomProviderINSTANCE.handleMap.tryGet(handle)
	if !ok {
		panic(fmt.Errorf("no callback in handle map: %d", handle))
	}

	result := make(chan C.UniffiForeignFutureResultRustBuffer, 1)
	cancel := make(chan struct{}, 1)
	guardHandle := cgo.NewHandle(cancel)
	*uniffiOutDroppedCallback = C.UniffiForeignFutureDroppedCallbackStruct{
		handle: C.uint64_t(guardHandle),
		free:   C.UniffiForeignFutureDroppedCallback(C.blazen_uniffiFreeGorutine),
	}

	// Wait for compleation or cancel
	go func() {
		select {
		case <-cancel:
		case res := <-result:
			C.call_UniffiForeignFutureCompleteRustBuffer(uniffiFutureCallback, uniffiCallbackData, res)
		}
	}()

	// Eval callback asynchroniously
	go func() {
		asyncResult := &C.UniffiForeignFutureResultRustBuffer{}
		uniffiOutReturn := &asyncResult.returnValue
		callStatus := &asyncResult.callStatus
		defer func() {
			result <- *asyncResult
		}()

		res, err :=
			uniffiObj.GenerateMusic(
				FfiConverterMusicRequestINSTANCE.Lift(GoRustBuffer{
					inner: request,
				}),
			)

		if err != nil {
			var actualError *BlazenError
			if errors.As(err, &actualError) {
				*callStatus = C.RustCallStatus{
					code:     C.int8_t(uniffiCallbackResultError),
					errorBuf: FfiConverterBlazenErrorINSTANCE.Lower(actualError),
				}
			} else {
				*callStatus = C.RustCallStatus{
					code: C.int8_t(uniffiCallbackUnexpectedResultError),
				}
			}
			return
		}

		*uniffiOutReturn = FfiConverterAudioResultINSTANCE.Lower(res)
	}()
}

//export blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod6
func blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod6(uniffiHandle C.uint64_t, request C.RustBuffer, uniffiFutureCallback C.UniffiForeignFutureCompleteRustBuffer, uniffiCallbackData C.uint64_t, uniffiOutDroppedCallback *C.UniffiForeignFutureDroppedCallbackStruct) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterCustomProviderINSTANCE.handleMap.tryGet(handle)
	if !ok {
		panic(fmt.Errorf("no callback in handle map: %d", handle))
	}

	result := make(chan C.UniffiForeignFutureResultRustBuffer, 1)
	cancel := make(chan struct{}, 1)
	guardHandle := cgo.NewHandle(cancel)
	*uniffiOutDroppedCallback = C.UniffiForeignFutureDroppedCallbackStruct{
		handle: C.uint64_t(guardHandle),
		free:   C.UniffiForeignFutureDroppedCallback(C.blazen_uniffiFreeGorutine),
	}

	// Wait for compleation or cancel
	go func() {
		select {
		case <-cancel:
		case res := <-result:
			C.call_UniffiForeignFutureCompleteRustBuffer(uniffiFutureCallback, uniffiCallbackData, res)
		}
	}()

	// Eval callback asynchroniously
	go func() {
		asyncResult := &C.UniffiForeignFutureResultRustBuffer{}
		uniffiOutReturn := &asyncResult.returnValue
		callStatus := &asyncResult.callStatus
		defer func() {
			result <- *asyncResult
		}()

		res, err :=
			uniffiObj.GenerateSfx(
				FfiConverterMusicRequestINSTANCE.Lift(GoRustBuffer{
					inner: request,
				}),
			)

		if err != nil {
			var actualError *BlazenError
			if errors.As(err, &actualError) {
				*callStatus = C.RustCallStatus{
					code:     C.int8_t(uniffiCallbackResultError),
					errorBuf: FfiConverterBlazenErrorINSTANCE.Lower(actualError),
				}
			} else {
				*callStatus = C.RustCallStatus{
					code: C.int8_t(uniffiCallbackUnexpectedResultError),
				}
			}
			return
		}

		*uniffiOutReturn = FfiConverterAudioResultINSTANCE.Lower(res)
	}()
}

//export blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod7
func blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod7(uniffiHandle C.uint64_t, request C.RustBuffer, uniffiFutureCallback C.UniffiForeignFutureCompleteRustBuffer, uniffiCallbackData C.uint64_t, uniffiOutDroppedCallback *C.UniffiForeignFutureDroppedCallbackStruct) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterCustomProviderINSTANCE.handleMap.tryGet(handle)
	if !ok {
		panic(fmt.Errorf("no callback in handle map: %d", handle))
	}

	result := make(chan C.UniffiForeignFutureResultRustBuffer, 1)
	cancel := make(chan struct{}, 1)
	guardHandle := cgo.NewHandle(cancel)
	*uniffiOutDroppedCallback = C.UniffiForeignFutureDroppedCallbackStruct{
		handle: C.uint64_t(guardHandle),
		free:   C.UniffiForeignFutureDroppedCallback(C.blazen_uniffiFreeGorutine),
	}

	// Wait for compleation or cancel
	go func() {
		select {
		case <-cancel:
		case res := <-result:
			C.call_UniffiForeignFutureCompleteRustBuffer(uniffiFutureCallback, uniffiCallbackData, res)
		}
	}()

	// Eval callback asynchroniously
	go func() {
		asyncResult := &C.UniffiForeignFutureResultRustBuffer{}
		uniffiOutReturn := &asyncResult.returnValue
		callStatus := &asyncResult.callStatus
		defer func() {
			result <- *asyncResult
		}()

		res, err :=
			uniffiObj.CloneVoice(
				FfiConverterVoiceCloneRequestINSTANCE.Lift(GoRustBuffer{
					inner: request,
				}),
			)

		if err != nil {
			var actualError *BlazenError
			if errors.As(err, &actualError) {
				*callStatus = C.RustCallStatus{
					code:     C.int8_t(uniffiCallbackResultError),
					errorBuf: FfiConverterBlazenErrorINSTANCE.Lower(actualError),
				}
			} else {
				*callStatus = C.RustCallStatus{
					code: C.int8_t(uniffiCallbackUnexpectedResultError),
				}
			}
			return
		}

		*uniffiOutReturn = FfiConverterVoiceHandleINSTANCE.Lower(res)
	}()
}

//export blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod8
func blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod8(uniffiHandle C.uint64_t, uniffiFutureCallback C.UniffiForeignFutureCompleteRustBuffer, uniffiCallbackData C.uint64_t, uniffiOutDroppedCallback *C.UniffiForeignFutureDroppedCallbackStruct) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterCustomProviderINSTANCE.handleMap.tryGet(handle)
	if !ok {
		panic(fmt.Errorf("no callback in handle map: %d", handle))
	}

	result := make(chan C.UniffiForeignFutureResultRustBuffer, 1)
	cancel := make(chan struct{}, 1)
	guardHandle := cgo.NewHandle(cancel)
	*uniffiOutDroppedCallback = C.UniffiForeignFutureDroppedCallbackStruct{
		handle: C.uint64_t(guardHandle),
		free:   C.UniffiForeignFutureDroppedCallback(C.blazen_uniffiFreeGorutine),
	}

	// Wait for compleation or cancel
	go func() {
		select {
		case <-cancel:
		case res := <-result:
			C.call_UniffiForeignFutureCompleteRustBuffer(uniffiFutureCallback, uniffiCallbackData, res)
		}
	}()

	// Eval callback asynchroniously
	go func() {
		asyncResult := &C.UniffiForeignFutureResultRustBuffer{}
		uniffiOutReturn := &asyncResult.returnValue
		callStatus := &asyncResult.callStatus
		defer func() {
			result <- *asyncResult
		}()

		res, err :=
			uniffiObj.ListVoices()

		if err != nil {
			var actualError *BlazenError
			if errors.As(err, &actualError) {
				*callStatus = C.RustCallStatus{
					code:     C.int8_t(uniffiCallbackResultError),
					errorBuf: FfiConverterBlazenErrorINSTANCE.Lower(actualError),
				}
			} else {
				*callStatus = C.RustCallStatus{
					code: C.int8_t(uniffiCallbackUnexpectedResultError),
				}
			}
			return
		}

		*uniffiOutReturn = FfiConverterSequenceVoiceHandleINSTANCE.Lower(res)
	}()
}

//export blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod9
func blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod9(uniffiHandle C.uint64_t, voice C.RustBuffer, uniffiFutureCallback C.UniffiForeignFutureCompleteVoid, uniffiCallbackData C.uint64_t, uniffiOutDroppedCallback *C.UniffiForeignFutureDroppedCallbackStruct) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterCustomProviderINSTANCE.handleMap.tryGet(handle)
	if !ok {
		panic(fmt.Errorf("no callback in handle map: %d", handle))
	}

	result := make(chan C.UniffiForeignFutureResultVoid, 1)
	cancel := make(chan struct{}, 1)
	guardHandle := cgo.NewHandle(cancel)
	*uniffiOutDroppedCallback = C.UniffiForeignFutureDroppedCallbackStruct{
		handle: C.uint64_t(guardHandle),
		free:   C.UniffiForeignFutureDroppedCallback(C.blazen_uniffiFreeGorutine),
	}

	// Wait for compleation or cancel
	go func() {
		select {
		case <-cancel:
		case res := <-result:
			C.call_UniffiForeignFutureCompleteVoid(uniffiFutureCallback, uniffiCallbackData, res)
		}
	}()

	// Eval callback asynchroniously
	go func() {
		asyncResult := &C.UniffiForeignFutureResultVoid{}
		callStatus := &asyncResult.callStatus
		defer func() {
			result <- *asyncResult
		}()

		err :=
			uniffiObj.DeleteVoice(
				FfiConverterVoiceHandleINSTANCE.Lift(GoRustBuffer{
					inner: voice,
				}),
			)

		if err != nil {
			var actualError *BlazenError
			if errors.As(err, &actualError) {
				*callStatus = C.RustCallStatus{
					code:     C.int8_t(uniffiCallbackResultError),
					errorBuf: FfiConverterBlazenErrorINSTANCE.Lower(actualError),
				}
			} else {
				*callStatus = C.RustCallStatus{
					code: C.int8_t(uniffiCallbackUnexpectedResultError),
				}
			}
			return
		}

	}()
}

//export blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod10
func blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod10(uniffiHandle C.uint64_t, request C.RustBuffer, uniffiFutureCallback C.UniffiForeignFutureCompleteRustBuffer, uniffiCallbackData C.uint64_t, uniffiOutDroppedCallback *C.UniffiForeignFutureDroppedCallbackStruct) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterCustomProviderINSTANCE.handleMap.tryGet(handle)
	if !ok {
		panic(fmt.Errorf("no callback in handle map: %d", handle))
	}

	result := make(chan C.UniffiForeignFutureResultRustBuffer, 1)
	cancel := make(chan struct{}, 1)
	guardHandle := cgo.NewHandle(cancel)
	*uniffiOutDroppedCallback = C.UniffiForeignFutureDroppedCallbackStruct{
		handle: C.uint64_t(guardHandle),
		free:   C.UniffiForeignFutureDroppedCallback(C.blazen_uniffiFreeGorutine),
	}

	// Wait for compleation or cancel
	go func() {
		select {
		case <-cancel:
		case res := <-result:
			C.call_UniffiForeignFutureCompleteRustBuffer(uniffiFutureCallback, uniffiCallbackData, res)
		}
	}()

	// Eval callback asynchroniously
	go func() {
		asyncResult := &C.UniffiForeignFutureResultRustBuffer{}
		uniffiOutReturn := &asyncResult.returnValue
		callStatus := &asyncResult.callStatus
		defer func() {
			result <- *asyncResult
		}()

		res, err :=
			uniffiObj.GenerateImage(
				FfiConverterImageRequestINSTANCE.Lift(GoRustBuffer{
					inner: request,
				}),
			)

		if err != nil {
			var actualError *BlazenError
			if errors.As(err, &actualError) {
				*callStatus = C.RustCallStatus{
					code:     C.int8_t(uniffiCallbackResultError),
					errorBuf: FfiConverterBlazenErrorINSTANCE.Lower(actualError),
				}
			} else {
				*callStatus = C.RustCallStatus{
					code: C.int8_t(uniffiCallbackUnexpectedResultError),
				}
			}
			return
		}

		*uniffiOutReturn = FfiConverterImageResultINSTANCE.Lower(res)
	}()
}

//export blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod11
func blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod11(uniffiHandle C.uint64_t, request C.RustBuffer, uniffiFutureCallback C.UniffiForeignFutureCompleteRustBuffer, uniffiCallbackData C.uint64_t, uniffiOutDroppedCallback *C.UniffiForeignFutureDroppedCallbackStruct) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterCustomProviderINSTANCE.handleMap.tryGet(handle)
	if !ok {
		panic(fmt.Errorf("no callback in handle map: %d", handle))
	}

	result := make(chan C.UniffiForeignFutureResultRustBuffer, 1)
	cancel := make(chan struct{}, 1)
	guardHandle := cgo.NewHandle(cancel)
	*uniffiOutDroppedCallback = C.UniffiForeignFutureDroppedCallbackStruct{
		handle: C.uint64_t(guardHandle),
		free:   C.UniffiForeignFutureDroppedCallback(C.blazen_uniffiFreeGorutine),
	}

	// Wait for compleation or cancel
	go func() {
		select {
		case <-cancel:
		case res := <-result:
			C.call_UniffiForeignFutureCompleteRustBuffer(uniffiFutureCallback, uniffiCallbackData, res)
		}
	}()

	// Eval callback asynchroniously
	go func() {
		asyncResult := &C.UniffiForeignFutureResultRustBuffer{}
		uniffiOutReturn := &asyncResult.returnValue
		callStatus := &asyncResult.callStatus
		defer func() {
			result <- *asyncResult
		}()

		res, err :=
			uniffiObj.UpscaleImage(
				FfiConverterUpscaleRequestINSTANCE.Lift(GoRustBuffer{
					inner: request,
				}),
			)

		if err != nil {
			var actualError *BlazenError
			if errors.As(err, &actualError) {
				*callStatus = C.RustCallStatus{
					code:     C.int8_t(uniffiCallbackResultError),
					errorBuf: FfiConverterBlazenErrorINSTANCE.Lower(actualError),
				}
			} else {
				*callStatus = C.RustCallStatus{
					code: C.int8_t(uniffiCallbackUnexpectedResultError),
				}
			}
			return
		}

		*uniffiOutReturn = FfiConverterImageResultINSTANCE.Lower(res)
	}()
}

//export blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod12
func blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod12(uniffiHandle C.uint64_t, request C.RustBuffer, uniffiFutureCallback C.UniffiForeignFutureCompleteRustBuffer, uniffiCallbackData C.uint64_t, uniffiOutDroppedCallback *C.UniffiForeignFutureDroppedCallbackStruct) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterCustomProviderINSTANCE.handleMap.tryGet(handle)
	if !ok {
		panic(fmt.Errorf("no callback in handle map: %d", handle))
	}

	result := make(chan C.UniffiForeignFutureResultRustBuffer, 1)
	cancel := make(chan struct{}, 1)
	guardHandle := cgo.NewHandle(cancel)
	*uniffiOutDroppedCallback = C.UniffiForeignFutureDroppedCallbackStruct{
		handle: C.uint64_t(guardHandle),
		free:   C.UniffiForeignFutureDroppedCallback(C.blazen_uniffiFreeGorutine),
	}

	// Wait for compleation or cancel
	go func() {
		select {
		case <-cancel:
		case res := <-result:
			C.call_UniffiForeignFutureCompleteRustBuffer(uniffiFutureCallback, uniffiCallbackData, res)
		}
	}()

	// Eval callback asynchroniously
	go func() {
		asyncResult := &C.UniffiForeignFutureResultRustBuffer{}
		uniffiOutReturn := &asyncResult.returnValue
		callStatus := &asyncResult.callStatus
		defer func() {
			result <- *asyncResult
		}()

		res, err :=
			uniffiObj.TextToVideo(
				FfiConverterVideoRequestINSTANCE.Lift(GoRustBuffer{
					inner: request,
				}),
			)

		if err != nil {
			var actualError *BlazenError
			if errors.As(err, &actualError) {
				*callStatus = C.RustCallStatus{
					code:     C.int8_t(uniffiCallbackResultError),
					errorBuf: FfiConverterBlazenErrorINSTANCE.Lower(actualError),
				}
			} else {
				*callStatus = C.RustCallStatus{
					code: C.int8_t(uniffiCallbackUnexpectedResultError),
				}
			}
			return
		}

		*uniffiOutReturn = FfiConverterVideoResultINSTANCE.Lower(res)
	}()
}

//export blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod13
func blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod13(uniffiHandle C.uint64_t, request C.RustBuffer, uniffiFutureCallback C.UniffiForeignFutureCompleteRustBuffer, uniffiCallbackData C.uint64_t, uniffiOutDroppedCallback *C.UniffiForeignFutureDroppedCallbackStruct) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterCustomProviderINSTANCE.handleMap.tryGet(handle)
	if !ok {
		panic(fmt.Errorf("no callback in handle map: %d", handle))
	}

	result := make(chan C.UniffiForeignFutureResultRustBuffer, 1)
	cancel := make(chan struct{}, 1)
	guardHandle := cgo.NewHandle(cancel)
	*uniffiOutDroppedCallback = C.UniffiForeignFutureDroppedCallbackStruct{
		handle: C.uint64_t(guardHandle),
		free:   C.UniffiForeignFutureDroppedCallback(C.blazen_uniffiFreeGorutine),
	}

	// Wait for compleation or cancel
	go func() {
		select {
		case <-cancel:
		case res := <-result:
			C.call_UniffiForeignFutureCompleteRustBuffer(uniffiFutureCallback, uniffiCallbackData, res)
		}
	}()

	// Eval callback asynchroniously
	go func() {
		asyncResult := &C.UniffiForeignFutureResultRustBuffer{}
		uniffiOutReturn := &asyncResult.returnValue
		callStatus := &asyncResult.callStatus
		defer func() {
			result <- *asyncResult
		}()

		res, err :=
			uniffiObj.ImageToVideo(
				FfiConverterVideoRequestINSTANCE.Lift(GoRustBuffer{
					inner: request,
				}),
			)

		if err != nil {
			var actualError *BlazenError
			if errors.As(err, &actualError) {
				*callStatus = C.RustCallStatus{
					code:     C.int8_t(uniffiCallbackResultError),
					errorBuf: FfiConverterBlazenErrorINSTANCE.Lower(actualError),
				}
			} else {
				*callStatus = C.RustCallStatus{
					code: C.int8_t(uniffiCallbackUnexpectedResultError),
				}
			}
			return
		}

		*uniffiOutReturn = FfiConverterVideoResultINSTANCE.Lower(res)
	}()
}

//export blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod14
func blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod14(uniffiHandle C.uint64_t, request C.RustBuffer, uniffiFutureCallback C.UniffiForeignFutureCompleteRustBuffer, uniffiCallbackData C.uint64_t, uniffiOutDroppedCallback *C.UniffiForeignFutureDroppedCallbackStruct) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterCustomProviderINSTANCE.handleMap.tryGet(handle)
	if !ok {
		panic(fmt.Errorf("no callback in handle map: %d", handle))
	}

	result := make(chan C.UniffiForeignFutureResultRustBuffer, 1)
	cancel := make(chan struct{}, 1)
	guardHandle := cgo.NewHandle(cancel)
	*uniffiOutDroppedCallback = C.UniffiForeignFutureDroppedCallbackStruct{
		handle: C.uint64_t(guardHandle),
		free:   C.UniffiForeignFutureDroppedCallback(C.blazen_uniffiFreeGorutine),
	}

	// Wait for compleation or cancel
	go func() {
		select {
		case <-cancel:
		case res := <-result:
			C.call_UniffiForeignFutureCompleteRustBuffer(uniffiFutureCallback, uniffiCallbackData, res)
		}
	}()

	// Eval callback asynchroniously
	go func() {
		asyncResult := &C.UniffiForeignFutureResultRustBuffer{}
		uniffiOutReturn := &asyncResult.returnValue
		callStatus := &asyncResult.callStatus
		defer func() {
			result <- *asyncResult
		}()

		res, err :=
			uniffiObj.Transcribe(
				FfiConverterTranscriptionRequestINSTANCE.Lift(GoRustBuffer{
					inner: request,
				}),
			)

		if err != nil {
			var actualError *BlazenError
			if errors.As(err, &actualError) {
				*callStatus = C.RustCallStatus{
					code:     C.int8_t(uniffiCallbackResultError),
					errorBuf: FfiConverterBlazenErrorINSTANCE.Lower(actualError),
				}
			} else {
				*callStatus = C.RustCallStatus{
					code: C.int8_t(uniffiCallbackUnexpectedResultError),
				}
			}
			return
		}

		*uniffiOutReturn = FfiConverterTranscriptionResultINSTANCE.Lower(res)
	}()
}

//export blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod15
func blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod15(uniffiHandle C.uint64_t, request C.RustBuffer, uniffiFutureCallback C.UniffiForeignFutureCompleteRustBuffer, uniffiCallbackData C.uint64_t, uniffiOutDroppedCallback *C.UniffiForeignFutureDroppedCallbackStruct) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterCustomProviderINSTANCE.handleMap.tryGet(handle)
	if !ok {
		panic(fmt.Errorf("no callback in handle map: %d", handle))
	}

	result := make(chan C.UniffiForeignFutureResultRustBuffer, 1)
	cancel := make(chan struct{}, 1)
	guardHandle := cgo.NewHandle(cancel)
	*uniffiOutDroppedCallback = C.UniffiForeignFutureDroppedCallbackStruct{
		handle: C.uint64_t(guardHandle),
		free:   C.UniffiForeignFutureDroppedCallback(C.blazen_uniffiFreeGorutine),
	}

	// Wait for compleation or cancel
	go func() {
		select {
		case <-cancel:
		case res := <-result:
			C.call_UniffiForeignFutureCompleteRustBuffer(uniffiFutureCallback, uniffiCallbackData, res)
		}
	}()

	// Eval callback asynchroniously
	go func() {
		asyncResult := &C.UniffiForeignFutureResultRustBuffer{}
		uniffiOutReturn := &asyncResult.returnValue
		callStatus := &asyncResult.callStatus
		defer func() {
			result <- *asyncResult
		}()

		res, err :=
			uniffiObj.Generate3d(
				FfiConverterThreeDRequestINSTANCE.Lift(GoRustBuffer{
					inner: request,
				}),
			)

		if err != nil {
			var actualError *BlazenError
			if errors.As(err, &actualError) {
				*callStatus = C.RustCallStatus{
					code:     C.int8_t(uniffiCallbackResultError),
					errorBuf: FfiConverterBlazenErrorINSTANCE.Lower(actualError),
				}
			} else {
				*callStatus = C.RustCallStatus{
					code: C.int8_t(uniffiCallbackUnexpectedResultError),
				}
			}
			return
		}

		*uniffiOutReturn = FfiConverterThreeDResultINSTANCE.Lower(res)
	}()
}

//export blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod16
func blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod16(uniffiHandle C.uint64_t, request C.RustBuffer, uniffiFutureCallback C.UniffiForeignFutureCompleteRustBuffer, uniffiCallbackData C.uint64_t, uniffiOutDroppedCallback *C.UniffiForeignFutureDroppedCallbackStruct) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterCustomProviderINSTANCE.handleMap.tryGet(handle)
	if !ok {
		panic(fmt.Errorf("no callback in handle map: %d", handle))
	}

	result := make(chan C.UniffiForeignFutureResultRustBuffer, 1)
	cancel := make(chan struct{}, 1)
	guardHandle := cgo.NewHandle(cancel)
	*uniffiOutDroppedCallback = C.UniffiForeignFutureDroppedCallbackStruct{
		handle: C.uint64_t(guardHandle),
		free:   C.UniffiForeignFutureDroppedCallback(C.blazen_uniffiFreeGorutine),
	}

	// Wait for compleation or cancel
	go func() {
		select {
		case <-cancel:
		case res := <-result:
			C.call_UniffiForeignFutureCompleteRustBuffer(uniffiFutureCallback, uniffiCallbackData, res)
		}
	}()

	// Eval callback asynchroniously
	go func() {
		asyncResult := &C.UniffiForeignFutureResultRustBuffer{}
		uniffiOutReturn := &asyncResult.returnValue
		callStatus := &asyncResult.callStatus
		defer func() {
			result <- *asyncResult
		}()

		res, err :=
			uniffiObj.RemoveBackground(
				FfiConverterBackgroundRemovalRequestINSTANCE.Lift(GoRustBuffer{
					inner: request,
				}),
			)

		if err != nil {
			var actualError *BlazenError
			if errors.As(err, &actualError) {
				*callStatus = C.RustCallStatus{
					code:     C.int8_t(uniffiCallbackResultError),
					errorBuf: FfiConverterBlazenErrorINSTANCE.Lower(actualError),
				}
			} else {
				*callStatus = C.RustCallStatus{
					code: C.int8_t(uniffiCallbackUnexpectedResultError),
				}
			}
			return
		}

		*uniffiOutReturn = FfiConverterImageResultINSTANCE.Lower(res)
	}()
}

var UniffiVTableCallbackInterfaceCustomProviderINSTANCE = C.UniffiVTableCallbackInterfaceCustomProvider{
	uniffiFree:       (C.UniffiCallbackInterfaceFree)(C.blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderFree),
	uniffiClone:      (C.UniffiCallbackInterfaceClone)(C.blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderClone),
	providerId:       (C.UniffiCallbackInterfaceCustomProviderMethod0)(C.blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod0),
	complete:         (C.UniffiCallbackInterfaceCustomProviderMethod1)(C.blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod1),
	stream:           (C.UniffiCallbackInterfaceCustomProviderMethod2)(C.blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod2),
	embed:            (C.UniffiCallbackInterfaceCustomProviderMethod3)(C.blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod3),
	textToSpeech:     (C.UniffiCallbackInterfaceCustomProviderMethod4)(C.blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod4),
	generateMusic:    (C.UniffiCallbackInterfaceCustomProviderMethod5)(C.blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod5),
	generateSfx:      (C.UniffiCallbackInterfaceCustomProviderMethod6)(C.blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod6),
	cloneVoice:       (C.UniffiCallbackInterfaceCustomProviderMethod7)(C.blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod7),
	listVoices:       (C.UniffiCallbackInterfaceCustomProviderMethod8)(C.blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod8),
	deleteVoice:      (C.UniffiCallbackInterfaceCustomProviderMethod9)(C.blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod9),
	generateImage:    (C.UniffiCallbackInterfaceCustomProviderMethod10)(C.blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod10),
	upscaleImage:     (C.UniffiCallbackInterfaceCustomProviderMethod11)(C.blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod11),
	textToVideo:      (C.UniffiCallbackInterfaceCustomProviderMethod12)(C.blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod12),
	imageToVideo:     (C.UniffiCallbackInterfaceCustomProviderMethod13)(C.blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod13),
	transcribe:       (C.UniffiCallbackInterfaceCustomProviderMethod14)(C.blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod14),
	generate3d:       (C.UniffiCallbackInterfaceCustomProviderMethod15)(C.blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod15),
	removeBackground: (C.UniffiCallbackInterfaceCustomProviderMethod16)(C.blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderMethod16),
}

//export blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderFree
func blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderFree(handle C.uint64_t) {
	FfiConverterCustomProviderINSTANCE.handleMap.remove(uint64(handle))
}

//export blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderClone
func blazen_uniffi_provider_custom_cgo_dispatchCallbackInterfaceCustomProviderClone(handle C.uint64_t) C.uint64_t {
	val, ok := FfiConverterCustomProviderINSTANCE.handleMap.tryGet(uint64(handle))
	if !ok {
		panic(fmt.Errorf("no callback in handle map: %d", handle))
	}
	return C.uint64_t(FfiConverterCustomProviderINSTANCE.handleMap.insert(val))
}

func (c FfiConverterCustomProvider) register() {
	C.uniffi_blazen_uniffi_fn_init_callback_vtable_customprovider(&UniffiVTableCallbackInterfaceCustomProviderINSTANCE)
}

// Opaque UniFFI handle that wraps the upstream
// [`blazen_llm::CustomProviderHandle`].
//
// Construct via one of the four free factory functions ([`ollama`],
// [`lm_studio`], [`openai_compat`], [`custom_provider_from_foreign`]). All
// 16 typed compute / completion methods dispatch through the inner handle,
// which applies any per-instance defaults attached via the builders before
// forwarding to the underlying [`CustomProvider`].
//
// The paired [`BaseProvider`] handle returned by [`as_base`](Self::as_base)
// exposes builder-style completion-defaults customisation
// (`with_system_prompt`, `with_tools_json`, ...).
type CustomProviderHandleInterface interface {
	// Return the paired [`BaseProvider`] handle for builder-style chaining.
	//
	// Use for `.with_system_prompt(...)`, `.with_tools_json(...)`,
	// `.with_response_format_json(...)`, or to hand the provider to an API
	// expecting an opaque `CompletionModel`-shaped handle.
	AsBase() *BaseProvider
	// Clone a voice from reference audio.
	CloneVoice(request VoiceCloneRequest) (VoiceHandle, error)
	// Perform a non-streaming chat completion. Applies any configured
	// completion defaults (system prompt, tools, response format) before
	// dispatching to the inner provider.
	Complete(request CompletionRequest) (CompletionResponse, error)
	// Delete a previously-cloned voice. Takes the voice id as a string so
	// foreign callers can pass `voice_handle.id` directly without
	// reconstructing the full record.
	DeleteVoice(voiceId string) (bool, error)
	// Embed one or more texts via the inner provider.
	Embed(texts []string) (EmbeddingResponse, error)
	// Generate a 3D model.
	Generate3d(request ThreeDRequest) (ThreeDResult, error)
	// Generate an image from a text prompt.
	GenerateImage(request ImageRequest) (ImageResult, error)
	// Generate music from a prompt.
	GenerateMusic(request MusicRequest) (AudioResult, error)
	// Generate sound effects from a prompt.
	GenerateSfx(request MusicRequest) (AudioResult, error)
	// Generate a video from a reference image.
	ImageToVideo(request VideoRequest) (VideoResult, error)
	// List voices known to the provider.
	ListVoices() ([]VoiceHandle, error)
	// The provider id of the wrapped inner provider.
	ProviderId() string
	// Remove the background from an existing image.
	RemoveBackground(request BackgroundRemovalRequest) (ImageResult, error)
	// Drive a streaming chat completion, dispatching each chunk to the sink.
	//
	// Symmetric with [`crate::streaming::complete_streaming`]: success and
	// failure are both delivered via the sink; the function itself only
	// returns `Err` if the initial request conversion fails.
	Stream(request CompletionRequest, sink CompletionStreamSink) error
	// Synthesize speech from text. Applies the configured speech defaults
	// (if any) before dispatching to the inner provider.
	TextToSpeech(request SpeechRequest) (AudioResult, error)
	// Generate a video from a text prompt.
	TextToVideo(request VideoRequest) (VideoResult, error)
	// Transcribe audio to text.
	Transcribe(request TranscriptionRequest) (TranscriptionResult, error)
	// Upscale an existing image.
	UpscaleImage(request UpscaleRequest) (ImageResult, error)
	// Set the default `response_format`. JSON-encoded `serde_json::Value`.
	//
	// Malformed JSON or an empty string is treated as JSON null (no default
	// response format).
	WithResponseFormatJson(fmtJson string) *CustomProviderHandle
	// Attach a default system prompt applied to every completion request
	// that doesn't already include a system message.
	//
	// Returns a fresh handle (clone-with-mutation) so the call composes
	// fluently with other `with_*` builders.
	WithSystemPrompt(prompt string) *CustomProviderHandle
	// Replace the default tool list. JSON-encoded `Vec<ToolDefinition>`.
	//
	// Malformed JSON or an empty string yields an empty tool list — matching
	// the permissive shape of the other `*_json` helpers on this type.
	WithToolsJson(toolsJson string) *CustomProviderHandle
}

// Opaque UniFFI handle that wraps the upstream
// [`blazen_llm::CustomProviderHandle`].
//
// Construct via one of the four free factory functions ([`ollama`],
// [`lm_studio`], [`openai_compat`], [`custom_provider_from_foreign`]). All
// 16 typed compute / completion methods dispatch through the inner handle,
// which applies any per-instance defaults attached via the builders before
// forwarding to the underlying [`CustomProvider`].
//
// The paired [`BaseProvider`] handle returned by [`as_base`](Self::as_base)
// exposes builder-style completion-defaults customisation
// (`with_system_prompt`, `with_tools_json`, ...).
type CustomProviderHandle struct {
	ffiObject FfiObject
}

// Return the paired [`BaseProvider`] handle for builder-style chaining.
//
// Use for `.with_system_prompt(...)`, `.with_tools_json(...)`,
// `.with_response_format_json(...)`, or to hand the provider to an API
// expecting an opaque `CompletionModel`-shaped handle.
func (_self *CustomProviderHandle) AsBase() *BaseProvider {
	_pointer := _self.ffiObject.incrementPointer("*CustomProviderHandle")
	defer _self.ffiObject.decrementPointer()
	return FfiConverterBaseProviderINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_method_customproviderhandle_as_base(
			_pointer, _uniffiStatus)
	}))
}

// Clone a voice from reference audio.
func (_self *CustomProviderHandle) CloneVoice(request VoiceCloneRequest) (VoiceHandle, error) {
	_pointer := _self.ffiObject.incrementPointer("*CustomProviderHandle")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) VoiceHandle {
			return FfiConverterVoiceHandleINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_customproviderhandle_clone_voice(
			_pointer, FfiConverterVoiceCloneRequestINSTANCE.Lower(request)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// Perform a non-streaming chat completion. Applies any configured
// completion defaults (system prompt, tools, response format) before
// dispatching to the inner provider.
func (_self *CustomProviderHandle) Complete(request CompletionRequest) (CompletionResponse, error) {
	_pointer := _self.ffiObject.incrementPointer("*CustomProviderHandle")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) CompletionResponse {
			return FfiConverterCompletionResponseINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_customproviderhandle_complete(
			_pointer, FfiConverterCompletionRequestINSTANCE.Lower(request)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// Delete a previously-cloned voice. Takes the voice id as a string so
// foreign callers can pass `voice_handle.id` directly without
// reconstructing the full record.
func (_self *CustomProviderHandle) DeleteVoice(voiceId string) (bool, error) {
	_pointer := _self.ffiObject.incrementPointer("*CustomProviderHandle")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) C.int8_t {
			res := C.ffi_blazen_uniffi_rust_future_complete_i8(handle, status)
			return res
		},
		// liftFn
		func(ffi C.int8_t) bool {
			return FfiConverterBoolINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_customproviderhandle_delete_voice(
			_pointer, FfiConverterStringINSTANCE.Lower(voiceId)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_i8(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_i8(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// Embed one or more texts via the inner provider.
func (_self *CustomProviderHandle) Embed(texts []string) (EmbeddingResponse, error) {
	_pointer := _self.ffiObject.incrementPointer("*CustomProviderHandle")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) EmbeddingResponse {
			return FfiConverterEmbeddingResponseINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_customproviderhandle_embed(
			_pointer, FfiConverterSequenceStringINSTANCE.Lower(texts)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// Generate a 3D model.
func (_self *CustomProviderHandle) Generate3d(request ThreeDRequest) (ThreeDResult, error) {
	_pointer := _self.ffiObject.incrementPointer("*CustomProviderHandle")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) ThreeDResult {
			return FfiConverterThreeDResultINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_customproviderhandle_generate_3d(
			_pointer, FfiConverterThreeDRequestINSTANCE.Lower(request)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// Generate an image from a text prompt.
func (_self *CustomProviderHandle) GenerateImage(request ImageRequest) (ImageResult, error) {
	_pointer := _self.ffiObject.incrementPointer("*CustomProviderHandle")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) ImageResult {
			return FfiConverterImageResultINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_customproviderhandle_generate_image(
			_pointer, FfiConverterImageRequestINSTANCE.Lower(request)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// Generate music from a prompt.
func (_self *CustomProviderHandle) GenerateMusic(request MusicRequest) (AudioResult, error) {
	_pointer := _self.ffiObject.incrementPointer("*CustomProviderHandle")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) AudioResult {
			return FfiConverterAudioResultINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_customproviderhandle_generate_music(
			_pointer, FfiConverterMusicRequestINSTANCE.Lower(request)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// Generate sound effects from a prompt.
func (_self *CustomProviderHandle) GenerateSfx(request MusicRequest) (AudioResult, error) {
	_pointer := _self.ffiObject.incrementPointer("*CustomProviderHandle")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) AudioResult {
			return FfiConverterAudioResultINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_customproviderhandle_generate_sfx(
			_pointer, FfiConverterMusicRequestINSTANCE.Lower(request)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// Generate a video from a reference image.
func (_self *CustomProviderHandle) ImageToVideo(request VideoRequest) (VideoResult, error) {
	_pointer := _self.ffiObject.incrementPointer("*CustomProviderHandle")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) VideoResult {
			return FfiConverterVideoResultINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_customproviderhandle_image_to_video(
			_pointer, FfiConverterVideoRequestINSTANCE.Lower(request)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// List voices known to the provider.
func (_self *CustomProviderHandle) ListVoices() ([]VoiceHandle, error) {
	_pointer := _self.ffiObject.incrementPointer("*CustomProviderHandle")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) []VoiceHandle {
			return FfiConverterSequenceVoiceHandleINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_customproviderhandle_list_voices(
			_pointer),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// The provider id of the wrapped inner provider.
func (_self *CustomProviderHandle) ProviderId() string {
	_pointer := _self.ffiObject.incrementPointer("*CustomProviderHandle")
	defer _self.ffiObject.decrementPointer()
	return FfiConverterStringINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) RustBufferI {
		return GoRustBuffer{
			inner: C.uniffi_blazen_uniffi_fn_method_customproviderhandle_provider_id(
				_pointer, _uniffiStatus),
		}
	}))
}

// Remove the background from an existing image.
func (_self *CustomProviderHandle) RemoveBackground(request BackgroundRemovalRequest) (ImageResult, error) {
	_pointer := _self.ffiObject.incrementPointer("*CustomProviderHandle")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) ImageResult {
			return FfiConverterImageResultINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_customproviderhandle_remove_background(
			_pointer, FfiConverterBackgroundRemovalRequestINSTANCE.Lower(request)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// Drive a streaming chat completion, dispatching each chunk to the sink.
//
// Symmetric with [`crate::streaming::complete_streaming`]: success and
// failure are both delivered via the sink; the function itself only
// returns `Err` if the initial request conversion fails.
func (_self *CustomProviderHandle) Stream(request CompletionRequest, sink CompletionStreamSink) error {
	_pointer := _self.ffiObject.incrementPointer("*CustomProviderHandle")
	defer _self.ffiObject.decrementPointer()
	_, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) struct{} {
			C.ffi_blazen_uniffi_rust_future_complete_void(handle, status)
			return struct{}{}
		},
		// liftFn
		func(_ struct{}) struct{} { return struct{}{} },
		C.uniffi_blazen_uniffi_fn_method_customproviderhandle_stream(
			_pointer, FfiConverterCompletionRequestINSTANCE.Lower(request), FfiConverterCompletionStreamSinkINSTANCE.Lower(sink)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_void(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_void(handle)
		},
	)

	if err == nil {
		return nil
	}

	return err
}

// Synthesize speech from text. Applies the configured speech defaults
// (if any) before dispatching to the inner provider.
func (_self *CustomProviderHandle) TextToSpeech(request SpeechRequest) (AudioResult, error) {
	_pointer := _self.ffiObject.incrementPointer("*CustomProviderHandle")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) AudioResult {
			return FfiConverterAudioResultINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_customproviderhandle_text_to_speech(
			_pointer, FfiConverterSpeechRequestINSTANCE.Lower(request)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// Generate a video from a text prompt.
func (_self *CustomProviderHandle) TextToVideo(request VideoRequest) (VideoResult, error) {
	_pointer := _self.ffiObject.incrementPointer("*CustomProviderHandle")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) VideoResult {
			return FfiConverterVideoResultINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_customproviderhandle_text_to_video(
			_pointer, FfiConverterVideoRequestINSTANCE.Lower(request)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// Transcribe audio to text.
func (_self *CustomProviderHandle) Transcribe(request TranscriptionRequest) (TranscriptionResult, error) {
	_pointer := _self.ffiObject.incrementPointer("*CustomProviderHandle")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) TranscriptionResult {
			return FfiConverterTranscriptionResultINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_customproviderhandle_transcribe(
			_pointer, FfiConverterTranscriptionRequestINSTANCE.Lower(request)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// Upscale an existing image.
func (_self *CustomProviderHandle) UpscaleImage(request UpscaleRequest) (ImageResult, error) {
	_pointer := _self.ffiObject.incrementPointer("*CustomProviderHandle")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) ImageResult {
			return FfiConverterImageResultINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_customproviderhandle_upscale_image(
			_pointer, FfiConverterUpscaleRequestINSTANCE.Lower(request)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// Set the default `response_format`. JSON-encoded `serde_json::Value`.
//
// Malformed JSON or an empty string is treated as JSON null (no default
// response format).
func (_self *CustomProviderHandle) WithResponseFormatJson(fmtJson string) *CustomProviderHandle {
	_pointer := _self.ffiObject.incrementPointer("*CustomProviderHandle")
	defer _self.ffiObject.decrementPointer()
	return FfiConverterCustomProviderHandleINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_method_customproviderhandle_with_response_format_json(
			_pointer, FfiConverterStringINSTANCE.Lower(fmtJson), _uniffiStatus)
	}))
}

// Attach a default system prompt applied to every completion request
// that doesn't already include a system message.
//
// Returns a fresh handle (clone-with-mutation) so the call composes
// fluently with other `with_*` builders.
func (_self *CustomProviderHandle) WithSystemPrompt(prompt string) *CustomProviderHandle {
	_pointer := _self.ffiObject.incrementPointer("*CustomProviderHandle")
	defer _self.ffiObject.decrementPointer()
	return FfiConverterCustomProviderHandleINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_method_customproviderhandle_with_system_prompt(
			_pointer, FfiConverterStringINSTANCE.Lower(prompt), _uniffiStatus)
	}))
}

// Replace the default tool list. JSON-encoded `Vec<ToolDefinition>`.
//
// Malformed JSON or an empty string yields an empty tool list — matching
// the permissive shape of the other `*_json` helpers on this type.
func (_self *CustomProviderHandle) WithToolsJson(toolsJson string) *CustomProviderHandle {
	_pointer := _self.ffiObject.incrementPointer("*CustomProviderHandle")
	defer _self.ffiObject.decrementPointer()
	return FfiConverterCustomProviderHandleINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_method_customproviderhandle_with_tools_json(
			_pointer, FfiConverterStringINSTANCE.Lower(toolsJson), _uniffiStatus)
	}))
}
func (object *CustomProviderHandle) Destroy() {
	runtime.SetFinalizer(object, nil)
	object.ffiObject.destroy()
}

type FfiConverterCustomProviderHandle struct{}

var FfiConverterCustomProviderHandleINSTANCE = FfiConverterCustomProviderHandle{}

func (c FfiConverterCustomProviderHandle) Lift(handle C.uint64_t) *CustomProviderHandle {
	result := &CustomProviderHandle{
		newFfiObject(
			handle,
			func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
				return C.uniffi_blazen_uniffi_fn_clone_customproviderhandle(handle, status)
			},
			func(handle C.uint64_t, status *C.RustCallStatus) {
				C.uniffi_blazen_uniffi_fn_free_customproviderhandle(handle, status)
			},
		),
	}
	runtime.SetFinalizer(result, (*CustomProviderHandle).Destroy)
	return result
}

func (c FfiConverterCustomProviderHandle) Read(reader io.Reader) *CustomProviderHandle {
	return c.Lift(C.uint64_t(readUint64(reader)))
}

func (c FfiConverterCustomProviderHandle) Lower(value *CustomProviderHandle) C.uint64_t {
	// TODO: this is bad - all synchronization from ObjectRuntime.go is discarded here,
	// because the handle will be decremented immediately after this function returns,
	// and someone will be left holding onto a non-locked handle.
	handle := value.ffiObject.incrementPointer("*CustomProviderHandle")
	defer value.ffiObject.decrementPointer()
	return handle
}

func (c FfiConverterCustomProviderHandle) Write(writer io.Writer, value *CustomProviderHandle) {
	writeUint64(writer, uint64(c.Lower(value)))
}

func LiftFromExternalCustomProviderHandle(handle uint64) *CustomProviderHandle {
	return FfiConverterCustomProviderHandleINSTANCE.Lift(C.uint64_t(handle))
}

func LowerToExternalCustomProviderHandle(value *CustomProviderHandle) uint64 {
	return uint64(FfiConverterCustomProviderHandleINSTANCE.Lower(value))
}

type FfiDestroyerCustomProviderHandle struct{}

func (_ FfiDestroyerCustomProviderHandle) Destroy(value *CustomProviderHandle) {
	value.Destroy()
}

// An embedding model that produces vector embeddings for text inputs.
//
// Construct one via the per-provider factories in `providers.rs`.
type EmbeddingModelInterface interface {
	// The dimensionality of vectors produced by this model.
	Dimensions() uint32
	// Embed one or more text strings, returning one vector per input.
	Embed(inputs []string) (EmbeddingResponse, error)
	// Synchronous variant of [`embed`](Self::embed).
	EmbedBlocking(inputs []string) (EmbeddingResponse, error)
	// The model's identifier (e.g. `"text-embedding-3-small"`).
	ModelId() string
}

// An embedding model that produces vector embeddings for text inputs.
//
// Construct one via the per-provider factories in `providers.rs`.
type EmbeddingModel struct {
	ffiObject FfiObject
}

// The dimensionality of vectors produced by this model.
func (_self *EmbeddingModel) Dimensions() uint32 {
	_pointer := _self.ffiObject.incrementPointer("*EmbeddingModel")
	defer _self.ffiObject.decrementPointer()
	return FfiConverterUint32INSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint32_t {
		return C.uniffi_blazen_uniffi_fn_method_embeddingmodel_dimensions(
			_pointer, _uniffiStatus)
	}))
}

// Embed one or more text strings, returning one vector per input.
func (_self *EmbeddingModel) Embed(inputs []string) (EmbeddingResponse, error) {
	_pointer := _self.ffiObject.incrementPointer("*EmbeddingModel")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) EmbeddingResponse {
			return FfiConverterEmbeddingResponseINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_embeddingmodel_embed(
			_pointer, FfiConverterSequenceStringINSTANCE.Lower(inputs)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// Synchronous variant of [`embed`](Self::embed).
func (_self *EmbeddingModel) EmbedBlocking(inputs []string) (EmbeddingResponse, error) {
	_pointer := _self.ffiObject.incrementPointer("*EmbeddingModel")
	defer _self.ffiObject.decrementPointer()
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) RustBufferI {
		return GoRustBuffer{
			inner: C.uniffi_blazen_uniffi_fn_method_embeddingmodel_embed_blocking(
				_pointer, FfiConverterSequenceStringINSTANCE.Lower(inputs), _uniffiStatus),
		}
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue EmbeddingResponse
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterEmbeddingResponseINSTANCE.Lift(_uniffiRV), nil
	}
}

// The model's identifier (e.g. `"text-embedding-3-small"`).
func (_self *EmbeddingModel) ModelId() string {
	_pointer := _self.ffiObject.incrementPointer("*EmbeddingModel")
	defer _self.ffiObject.decrementPointer()
	return FfiConverterStringINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) RustBufferI {
		return GoRustBuffer{
			inner: C.uniffi_blazen_uniffi_fn_method_embeddingmodel_model_id(
				_pointer, _uniffiStatus),
		}
	}))
}
func (object *EmbeddingModel) Destroy() {
	runtime.SetFinalizer(object, nil)
	object.ffiObject.destroy()
}

type FfiConverterEmbeddingModel struct{}

var FfiConverterEmbeddingModelINSTANCE = FfiConverterEmbeddingModel{}

func (c FfiConverterEmbeddingModel) Lift(handle C.uint64_t) *EmbeddingModel {
	result := &EmbeddingModel{
		newFfiObject(
			handle,
			func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
				return C.uniffi_blazen_uniffi_fn_clone_embeddingmodel(handle, status)
			},
			func(handle C.uint64_t, status *C.RustCallStatus) {
				C.uniffi_blazen_uniffi_fn_free_embeddingmodel(handle, status)
			},
		),
	}
	runtime.SetFinalizer(result, (*EmbeddingModel).Destroy)
	return result
}

func (c FfiConverterEmbeddingModel) Read(reader io.Reader) *EmbeddingModel {
	return c.Lift(C.uint64_t(readUint64(reader)))
}

func (c FfiConverterEmbeddingModel) Lower(value *EmbeddingModel) C.uint64_t {
	// TODO: this is bad - all synchronization from ObjectRuntime.go is discarded here,
	// because the handle will be decremented immediately after this function returns,
	// and someone will be left holding onto a non-locked handle.
	handle := value.ffiObject.incrementPointer("*EmbeddingModel")
	defer value.ffiObject.decrementPointer()
	return handle
}

func (c FfiConverterEmbeddingModel) Write(writer io.Writer, value *EmbeddingModel) {
	writeUint64(writer, uint64(c.Lower(value)))
}

func LiftFromExternalEmbeddingModel(handle uint64) *EmbeddingModel {
	return FfiConverterEmbeddingModelINSTANCE.Lift(C.uint64_t(handle))
}

func LowerToExternalEmbeddingModel(value *EmbeddingModel) uint64 {
	return uint64(FfiConverterEmbeddingModelINSTANCE.Lower(value))
}

type FfiDestroyerEmbeddingModel struct{}

func (_ FfiDestroyerEmbeddingModel) Destroy(value *EmbeddingModel) {
	value.Destroy()
}

// An image-generation model.
//
// Construct via [`new_diffusion_model`] (local, feature-gated) or
// [`new_fal_image_gen_model`] (cloud). Once obtained, call
// [`generate`](Self::generate) (async) or
// [`generate_blocking`](Self::generate_blocking) (sync) to render images.
type ImageGenModelInterface interface {
	// Generate `num_images` images for `prompt` at the given dimensions.
	//
	// `negative_prompt` describes content to avoid; `model` overrides the
	// provider's default endpoint (e.g. a specific fal.ai model id).
	// Backends ignore knobs they don't support.
	Generate(prompt string, negativePrompt *string, width *uint32, height *uint32, numImages *uint32, model *string) (ImageGenResult, error)
	// Synchronous variant of [`generate`](Self::generate).
	GenerateBlocking(prompt string, negativePrompt *string, width *uint32, height *uint32, numImages *uint32, model *string) (ImageGenResult, error)
}

// An image-generation model.
//
// Construct via [`new_diffusion_model`] (local, feature-gated) or
// [`new_fal_image_gen_model`] (cloud). Once obtained, call
// [`generate`](Self::generate) (async) or
// [`generate_blocking`](Self::generate_blocking) (sync) to render images.
type ImageGenModel struct {
	ffiObject FfiObject
}

// Generate `num_images` images for `prompt` at the given dimensions.
//
// `negative_prompt` describes content to avoid; `model` overrides the
// provider's default endpoint (e.g. a specific fal.ai model id).
// Backends ignore knobs they don't support.
func (_self *ImageGenModel) Generate(prompt string, negativePrompt *string, width *uint32, height *uint32, numImages *uint32, model *string) (ImageGenResult, error) {
	_pointer := _self.ffiObject.incrementPointer("*ImageGenModel")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) ImageGenResult {
			return FfiConverterImageGenResultINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_imagegenmodel_generate(
			_pointer, FfiConverterStringINSTANCE.Lower(prompt), FfiConverterOptionalStringINSTANCE.Lower(negativePrompt), FfiConverterOptionalUint32INSTANCE.Lower(width), FfiConverterOptionalUint32INSTANCE.Lower(height), FfiConverterOptionalUint32INSTANCE.Lower(numImages), FfiConverterOptionalStringINSTANCE.Lower(model)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// Synchronous variant of [`generate`](Self::generate).
func (_self *ImageGenModel) GenerateBlocking(prompt string, negativePrompt *string, width *uint32, height *uint32, numImages *uint32, model *string) (ImageGenResult, error) {
	_pointer := _self.ffiObject.incrementPointer("*ImageGenModel")
	defer _self.ffiObject.decrementPointer()
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) RustBufferI {
		return GoRustBuffer{
			inner: C.uniffi_blazen_uniffi_fn_method_imagegenmodel_generate_blocking(
				_pointer, FfiConverterStringINSTANCE.Lower(prompt), FfiConverterOptionalStringINSTANCE.Lower(negativePrompt), FfiConverterOptionalUint32INSTANCE.Lower(width), FfiConverterOptionalUint32INSTANCE.Lower(height), FfiConverterOptionalUint32INSTANCE.Lower(numImages), FfiConverterOptionalStringINSTANCE.Lower(model), _uniffiStatus),
		}
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue ImageGenResult
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterImageGenResultINSTANCE.Lift(_uniffiRV), nil
	}
}
func (object *ImageGenModel) Destroy() {
	runtime.SetFinalizer(object, nil)
	object.ffiObject.destroy()
}

type FfiConverterImageGenModel struct{}

var FfiConverterImageGenModelINSTANCE = FfiConverterImageGenModel{}

func (c FfiConverterImageGenModel) Lift(handle C.uint64_t) *ImageGenModel {
	result := &ImageGenModel{
		newFfiObject(
			handle,
			func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
				return C.uniffi_blazen_uniffi_fn_clone_imagegenmodel(handle, status)
			},
			func(handle C.uint64_t, status *C.RustCallStatus) {
				C.uniffi_blazen_uniffi_fn_free_imagegenmodel(handle, status)
			},
		),
	}
	runtime.SetFinalizer(result, (*ImageGenModel).Destroy)
	return result
}

func (c FfiConverterImageGenModel) Read(reader io.Reader) *ImageGenModel {
	return c.Lift(C.uint64_t(readUint64(reader)))
}

func (c FfiConverterImageGenModel) Lower(value *ImageGenModel) C.uint64_t {
	// TODO: this is bad - all synchronization from ObjectRuntime.go is discarded here,
	// because the handle will be decremented immediately after this function returns,
	// and someone will be left holding onto a non-locked handle.
	handle := value.ffiObject.incrementPointer("*ImageGenModel")
	defer value.ffiObject.decrementPointer()
	return handle
}

func (c FfiConverterImageGenModel) Write(writer io.Writer, value *ImageGenModel) {
	writeUint64(writer, uint64(c.Lower(value)))
}

func LiftFromExternalImageGenModel(handle uint64) *ImageGenModel {
	return FfiConverterImageGenModelINSTANCE.Lift(C.uint64_t(handle))
}

func LowerToExternalImageGenModel(value *ImageGenModel) uint64 {
	return uint64(FfiConverterImageGenModelINSTANCE.Lower(value))
}

type FfiDestroyerImageGenModel struct{}

func (_ FfiDestroyerImageGenModel) Destroy(value *ImageGenModel) {
	value.Destroy()
}

// Client handle for invoking workflows on a remote [`PeerServer`].
//
// Construct with [`PeerClient::connect`]. RPCs go out over a multiplexed
// HTTP/2 channel held inside the client; multiple concurrent calls on the
// same `PeerClient` are safe and share the connection.
type PeerClientInterface interface {
	// The node-id stamped into outgoing requests for tracing.
	NodeId() string
	// Invoke a workflow on the connected peer and wait for its terminal
	// result.
	//
	// - `workflow_name` is the symbolic name the remote peer's
	// [`blazen_core::step_registry`] knows the workflow as.
	// - `step_ids` is the ordered list of step identifiers to execute.
	// Every entry must be registered on the remote peer's process or
	// the call fails with [`BlazenError::Peer`] (`kind = "UnknownStep"`).
	// This is required by the peer wire protocol — see
	// [`blazen_peer::SubWorkflowRequest`].
	// - `input_json` is the JSON-encoded payload fed into the workflow's
	// entry step.
	// - `timeout_secs` bounds the remote workflow's wall-clock execution.
	// `None` defers to the server's default deadline.
	//
	// The returned [`WorkflowResult`] carries the terminal `StopEvent`
	// payload synthesised from the remote `SubWorkflowResponse`. Per-run
	// LLM token usage and cost are not propagated over the wire and are
	// reported as zero; foreign callers needing those should query the
	// remote peer's telemetry directly.
	//
	// # Errors
	//
	// Returns [`BlazenError::Peer`] for encode / transport / envelope-
	// version failures, or [`BlazenError::Workflow`] when the remote
	// reports a workflow-execution error in `SubWorkflowResponse::error`.
	RunRemoteWorkflow(workflowName string, stepIds []string, inputJson string, timeoutSecs *uint64) (WorkflowResult, error)
	// Synchronous variant of [`PeerClient::run_remote_workflow`] — blocks
	// the current thread on the shared Tokio runtime.
	//
	// # Errors
	//
	// Same as [`PeerClient::run_remote_workflow`].
	RunRemoteWorkflowBlocking(workflowName string, stepIds []string, inputJson string, timeoutSecs *uint64) (WorkflowResult, error)
}

// Client handle for invoking workflows on a remote [`PeerServer`].
//
// Construct with [`PeerClient::connect`]. RPCs go out over a multiplexed
// HTTP/2 channel held inside the client; multiple concurrent calls on the
// same `PeerClient` are safe and share the connection.
type PeerClient struct {
	ffiObject FfiObject
}

// Open a connection to the peer at `address`.
//
// `address` must be a valid gRPC endpoint URI such as
// `"http://node-a.local:7443"`. `client_node_id` identifies *this* end
// of the connection in trace logs on both sides and is typically the
// local hostname or a process-startup UUID.
//
// This constructor is blocking — it drives the TCP connect on the
// shared Tokio runtime so foreign callers without an async story
// (Ruby, synchronous Go code) can still set up a client. The async
// connect path is internal to upstream `BlazenPeerClient` and is not
// re-exposed across UniFFI to avoid a constructor that returns a
// coroutine in every target language.
//
// # Errors
//
// Returns [`BlazenError::Peer`] (`kind = "Transport"`) if the
// endpoint URI is invalid or the TCP / HTTP/2 handshake fails.
func PeerClientConnect(address string, clientNodeId string) (*PeerClient, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_constructor_peerclient_connect(FfiConverterStringINSTANCE.Lower(address), FfiConverterStringINSTANCE.Lower(clientNodeId), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *PeerClient
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterPeerClientINSTANCE.Lift(_uniffiRV), nil
	}
}

// The node-id stamped into outgoing requests for tracing.
func (_self *PeerClient) NodeId() string {
	_pointer := _self.ffiObject.incrementPointer("*PeerClient")
	defer _self.ffiObject.decrementPointer()
	return FfiConverterStringINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) RustBufferI {
		return GoRustBuffer{
			inner: C.uniffi_blazen_uniffi_fn_method_peerclient_node_id(
				_pointer, _uniffiStatus),
		}
	}))
}

// Invoke a workflow on the connected peer and wait for its terminal
// result.
//
// - `workflow_name` is the symbolic name the remote peer's
// [`blazen_core::step_registry`] knows the workflow as.
// - `step_ids` is the ordered list of step identifiers to execute.
// Every entry must be registered on the remote peer's process or
// the call fails with [`BlazenError::Peer`] (`kind = "UnknownStep"`).
// This is required by the peer wire protocol — see
// [`blazen_peer::SubWorkflowRequest`].
// - `input_json` is the JSON-encoded payload fed into the workflow's
// entry step.
// - `timeout_secs` bounds the remote workflow's wall-clock execution.
// `None` defers to the server's default deadline.
//
// The returned [`WorkflowResult`] carries the terminal `StopEvent`
// payload synthesised from the remote `SubWorkflowResponse`. Per-run
// LLM token usage and cost are not propagated over the wire and are
// reported as zero; foreign callers needing those should query the
// remote peer's telemetry directly.
//
// # Errors
//
// Returns [`BlazenError::Peer`] for encode / transport / envelope-
// version failures, or [`BlazenError::Workflow`] when the remote
// reports a workflow-execution error in `SubWorkflowResponse::error`.
func (_self *PeerClient) RunRemoteWorkflow(workflowName string, stepIds []string, inputJson string, timeoutSecs *uint64) (WorkflowResult, error) {
	_pointer := _self.ffiObject.incrementPointer("*PeerClient")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) WorkflowResult {
			return FfiConverterWorkflowResultINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_peerclient_run_remote_workflow(
			_pointer, FfiConverterStringINSTANCE.Lower(workflowName), FfiConverterSequenceStringINSTANCE.Lower(stepIds), FfiConverterStringINSTANCE.Lower(inputJson), FfiConverterOptionalUint64INSTANCE.Lower(timeoutSecs)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// Synchronous variant of [`PeerClient::run_remote_workflow`] — blocks
// the current thread on the shared Tokio runtime.
//
// # Errors
//
// Same as [`PeerClient::run_remote_workflow`].
func (_self *PeerClient) RunRemoteWorkflowBlocking(workflowName string, stepIds []string, inputJson string, timeoutSecs *uint64) (WorkflowResult, error) {
	_pointer := _self.ffiObject.incrementPointer("*PeerClient")
	defer _self.ffiObject.decrementPointer()
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) RustBufferI {
		return GoRustBuffer{
			inner: C.uniffi_blazen_uniffi_fn_method_peerclient_run_remote_workflow_blocking(
				_pointer, FfiConverterStringINSTANCE.Lower(workflowName), FfiConverterSequenceStringINSTANCE.Lower(stepIds), FfiConverterStringINSTANCE.Lower(inputJson), FfiConverterOptionalUint64INSTANCE.Lower(timeoutSecs), _uniffiStatus),
		}
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue WorkflowResult
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterWorkflowResultINSTANCE.Lift(_uniffiRV), nil
	}
}
func (object *PeerClient) Destroy() {
	runtime.SetFinalizer(object, nil)
	object.ffiObject.destroy()
}

type FfiConverterPeerClient struct{}

var FfiConverterPeerClientINSTANCE = FfiConverterPeerClient{}

func (c FfiConverterPeerClient) Lift(handle C.uint64_t) *PeerClient {
	result := &PeerClient{
		newFfiObject(
			handle,
			func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
				return C.uniffi_blazen_uniffi_fn_clone_peerclient(handle, status)
			},
			func(handle C.uint64_t, status *C.RustCallStatus) {
				C.uniffi_blazen_uniffi_fn_free_peerclient(handle, status)
			},
		),
	}
	runtime.SetFinalizer(result, (*PeerClient).Destroy)
	return result
}

func (c FfiConverterPeerClient) Read(reader io.Reader) *PeerClient {
	return c.Lift(C.uint64_t(readUint64(reader)))
}

func (c FfiConverterPeerClient) Lower(value *PeerClient) C.uint64_t {
	// TODO: this is bad - all synchronization from ObjectRuntime.go is discarded here,
	// because the handle will be decremented immediately after this function returns,
	// and someone will be left holding onto a non-locked handle.
	handle := value.ffiObject.incrementPointer("*PeerClient")
	defer value.ffiObject.decrementPointer()
	return handle
}

func (c FfiConverterPeerClient) Write(writer io.Writer, value *PeerClient) {
	writeUint64(writer, uint64(c.Lower(value)))
}

func LiftFromExternalPeerClient(handle uint64) *PeerClient {
	return FfiConverterPeerClientINSTANCE.Lift(C.uint64_t(handle))
}

func LowerToExternalPeerClient(value *PeerClient) uint64 {
	return uint64(FfiConverterPeerClientINSTANCE.Lower(value))
}

type FfiDestroyerPeerClient struct{}

func (_ FfiDestroyerPeerClient) Destroy(value *PeerClient) {
	value.Destroy()
}

// Node-local Blazen peer gRPC server.
//
// Owns a stable `node_id` embedded in every
// `RemoteRefDescriptor` this peer hands out, plus an in-process session-ref
// registry. Construct with [`PeerServer::new`] and start the gRPC listener
// with [`PeerServer::serve`] (async) or [`PeerServer::serve_blocking`].
//
// Dispatched workflows are resolved at request time through the
// process-wide [`blazen_core::step_registry`], so any workflow whose steps
// have been registered in this process can be invoked remotely by name.
type PeerServerInterface interface {
	// Bind the gRPC server to `listen_address` and serve until the
	// listener errors or the caller's async task is cancelled.
	//
	// `listen_address` must parse as a [`std::net::SocketAddr`] (for
	// example `"0.0.0.0:50051"` or `"127.0.0.1:7443"`). This method
	// consumes the underlying server; calling it twice on the same
	// `PeerServer` returns [`BlazenError::Validation`].
	//
	// # Errors
	//
	// Returns [`BlazenError::Validation`] if `listen_address` cannot be
	// parsed or the server has already been consumed by a prior call, and
	// [`BlazenError::Peer`] (`kind = "Transport"`) if the gRPC listener
	// fails to bind or encounters a fatal I/O error while serving.
	Serve(listenAddress string) error
	// Synchronous variant of [`PeerServer::serve`] — blocks the current
	// thread on the shared Tokio runtime until the server exits. Intended
	// for foreign callers (Ruby scripts, Go `main`, Swift CLIs) that want
	// a one-shot blocking bind without driving an async runtime.
	//
	// # Errors
	//
	// Same as [`PeerServer::serve`].
	ServeBlocking(listenAddress string) error
}

// Node-local Blazen peer gRPC server.
//
// Owns a stable `node_id` embedded in every
// `RemoteRefDescriptor` this peer hands out, plus an in-process session-ref
// registry. Construct with [`PeerServer::new`] and start the gRPC listener
// with [`PeerServer::serve`] (async) or [`PeerServer::serve_blocking`].
//
// Dispatched workflows are resolved at request time through the
// process-wide [`blazen_core::step_registry`], so any workflow whose steps
// have been registered in this process can be invoked remotely by name.
type PeerServer struct {
	ffiObject FfiObject
}

// Create a new peer server with a fresh, empty session-ref registry.
//
// `node_id` is the stable identifier that this server stamps onto every
// `RemoteRefDescriptor` it returns. Typical values are the hostname or
// a UUID picked at process startup.
func NewPeerServer(nodeId string) *PeerServer {
	return FfiConverterPeerServerINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_constructor_peerserver_new(FfiConverterStringINSTANCE.Lower(nodeId), _uniffiStatus)
	}))
}

// Bind the gRPC server to `listen_address` and serve until the
// listener errors or the caller's async task is cancelled.
//
// `listen_address` must parse as a [`std::net::SocketAddr`] (for
// example `"0.0.0.0:50051"` or `"127.0.0.1:7443"`). This method
// consumes the underlying server; calling it twice on the same
// `PeerServer` returns [`BlazenError::Validation`].
//
// # Errors
//
// Returns [`BlazenError::Validation`] if `listen_address` cannot be
// parsed or the server has already been consumed by a prior call, and
// [`BlazenError::Peer`] (`kind = "Transport"`) if the gRPC listener
// fails to bind or encounters a fatal I/O error while serving.
func (_self *PeerServer) Serve(listenAddress string) error {
	_pointer := _self.ffiObject.incrementPointer("*PeerServer")
	defer _self.ffiObject.decrementPointer()
	_, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) struct{} {
			C.ffi_blazen_uniffi_rust_future_complete_void(handle, status)
			return struct{}{}
		},
		// liftFn
		func(_ struct{}) struct{} { return struct{}{} },
		C.uniffi_blazen_uniffi_fn_method_peerserver_serve(
			_pointer, FfiConverterStringINSTANCE.Lower(listenAddress)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_void(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_void(handle)
		},
	)

	if err == nil {
		return nil
	}

	return err
}

// Synchronous variant of [`PeerServer::serve`] — blocks the current
// thread on the shared Tokio runtime until the server exits. Intended
// for foreign callers (Ruby scripts, Go `main`, Swift CLIs) that want
// a one-shot blocking bind without driving an async runtime.
//
// # Errors
//
// Same as [`PeerServer::serve`].
func (_self *PeerServer) ServeBlocking(listenAddress string) error {
	_pointer := _self.ffiObject.incrementPointer("*PeerServer")
	defer _self.ffiObject.decrementPointer()
	_, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) bool {
		C.uniffi_blazen_uniffi_fn_method_peerserver_serve_blocking(
			_pointer, FfiConverterStringINSTANCE.Lower(listenAddress), _uniffiStatus)
		return false
	})
	return _uniffiErr.AsError()
}
func (object *PeerServer) Destroy() {
	runtime.SetFinalizer(object, nil)
	object.ffiObject.destroy()
}

type FfiConverterPeerServer struct{}

var FfiConverterPeerServerINSTANCE = FfiConverterPeerServer{}

func (c FfiConverterPeerServer) Lift(handle C.uint64_t) *PeerServer {
	result := &PeerServer{
		newFfiObject(
			handle,
			func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
				return C.uniffi_blazen_uniffi_fn_clone_peerserver(handle, status)
			},
			func(handle C.uint64_t, status *C.RustCallStatus) {
				C.uniffi_blazen_uniffi_fn_free_peerserver(handle, status)
			},
		),
	}
	runtime.SetFinalizer(result, (*PeerServer).Destroy)
	return result
}

func (c FfiConverterPeerServer) Read(reader io.Reader) *PeerServer {
	return c.Lift(C.uint64_t(readUint64(reader)))
}

func (c FfiConverterPeerServer) Lower(value *PeerServer) C.uint64_t {
	// TODO: this is bad - all synchronization from ObjectRuntime.go is discarded here,
	// because the handle will be decremented immediately after this function returns,
	// and someone will be left holding onto a non-locked handle.
	handle := value.ffiObject.incrementPointer("*PeerServer")
	defer value.ffiObject.decrementPointer()
	return handle
}

func (c FfiConverterPeerServer) Write(writer io.Writer, value *PeerServer) {
	writeUint64(writer, uint64(c.Lower(value)))
}

func LiftFromExternalPeerServer(handle uint64) *PeerServer {
	return FfiConverterPeerServerINSTANCE.Lift(C.uint64_t(handle))
}

func LowerToExternalPeerServer(value *PeerServer) uint64 {
	return uint64(FfiConverterPeerServerINSTANCE.Lower(value))
}

type FfiDestroyerPeerServer struct{}

func (_ FfiDestroyerPeerServer) Destroy(value *PeerServer) {
	value.Destroy()
}

// A validated, runnable pipeline.
//
// Multiple runs are allowed — invoking [`run`](Self::run) twice in a row
// is safe and produces independent runs — but the implementation rejects
// **overlapping** runs on the same handle to avoid surprising aliasing of
// inner workflow state across concurrent foreign callers.
type PipelineInterface interface {
	// Execute the pipeline to completion. `input_json` is parsed as JSON
	// and passed as the first stage's `StartEvent` payload; each
	// subsequent stage receives the previous stage's `StopEvent` result.
	//
	// Returns a [`WorkflowResult`] whose `event` field is a synthetic
	// `StopEvent` carrying the final stage output, and whose
	// `total_*_tokens` / `total_cost_usd` fields are the sum across every
	// stage's `WorkflowResult`.
	Run(inputJson string) (WorkflowResult, error)
	// Synchronous variant of [`run`](Self::run) — blocks the current
	// thread on the shared Tokio runtime. Provided for callers that want
	// fire-and-forget usage without engaging their host language's async
	// machinery (Ruby scripts, simple Go `main` functions).
	RunBlocking(inputJson string) (WorkflowResult, error)
	// Stage names in registration order — useful for foreign-side
	// introspection / debug logging without re-running the pipeline.
	StageNames() []string
}

// A validated, runnable pipeline.
//
// Multiple runs are allowed — invoking [`run`](Self::run) twice in a row
// is safe and produces independent runs — but the implementation rejects
// **overlapping** runs on the same handle to avoid surprising aliasing of
// inner workflow state across concurrent foreign callers.
type Pipeline struct {
	ffiObject FfiObject
}

// Execute the pipeline to completion. `input_json` is parsed as JSON
// and passed as the first stage's `StartEvent` payload; each
// subsequent stage receives the previous stage's `StopEvent` result.
//
// Returns a [`WorkflowResult`] whose `event` field is a synthetic
// `StopEvent` carrying the final stage output, and whose
// `total_*_tokens` / `total_cost_usd` fields are the sum across every
// stage's `WorkflowResult`.
func (_self *Pipeline) Run(inputJson string) (WorkflowResult, error) {
	_pointer := _self.ffiObject.incrementPointer("*Pipeline")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) WorkflowResult {
			return FfiConverterWorkflowResultINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_pipeline_run(
			_pointer, FfiConverterStringINSTANCE.Lower(inputJson)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// Synchronous variant of [`run`](Self::run) — blocks the current
// thread on the shared Tokio runtime. Provided for callers that want
// fire-and-forget usage without engaging their host language's async
// machinery (Ruby scripts, simple Go `main` functions).
func (_self *Pipeline) RunBlocking(inputJson string) (WorkflowResult, error) {
	_pointer := _self.ffiObject.incrementPointer("*Pipeline")
	defer _self.ffiObject.decrementPointer()
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) RustBufferI {
		return GoRustBuffer{
			inner: C.uniffi_blazen_uniffi_fn_method_pipeline_run_blocking(
				_pointer, FfiConverterStringINSTANCE.Lower(inputJson), _uniffiStatus),
		}
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue WorkflowResult
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterWorkflowResultINSTANCE.Lift(_uniffiRV), nil
	}
}

// Stage names in registration order — useful for foreign-side
// introspection / debug logging without re-running the pipeline.
func (_self *Pipeline) StageNames() []string {
	_pointer := _self.ffiObject.incrementPointer("*Pipeline")
	defer _self.ffiObject.decrementPointer()
	return FfiConverterSequenceStringINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) RustBufferI {
		return GoRustBuffer{
			inner: C.uniffi_blazen_uniffi_fn_method_pipeline_stage_names(
				_pointer, _uniffiStatus),
		}
	}))
}
func (object *Pipeline) Destroy() {
	runtime.SetFinalizer(object, nil)
	object.ffiObject.destroy()
}

type FfiConverterPipeline struct{}

var FfiConverterPipelineINSTANCE = FfiConverterPipeline{}

func (c FfiConverterPipeline) Lift(handle C.uint64_t) *Pipeline {
	result := &Pipeline{
		newFfiObject(
			handle,
			func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
				return C.uniffi_blazen_uniffi_fn_clone_pipeline(handle, status)
			},
			func(handle C.uint64_t, status *C.RustCallStatus) {
				C.uniffi_blazen_uniffi_fn_free_pipeline(handle, status)
			},
		),
	}
	runtime.SetFinalizer(result, (*Pipeline).Destroy)
	return result
}

func (c FfiConverterPipeline) Read(reader io.Reader) *Pipeline {
	return c.Lift(C.uint64_t(readUint64(reader)))
}

func (c FfiConverterPipeline) Lower(value *Pipeline) C.uint64_t {
	// TODO: this is bad - all synchronization from ObjectRuntime.go is discarded here,
	// because the handle will be decremented immediately after this function returns,
	// and someone will be left holding onto a non-locked handle.
	handle := value.ffiObject.incrementPointer("*Pipeline")
	defer value.ffiObject.decrementPointer()
	return handle
}

func (c FfiConverterPipeline) Write(writer io.Writer, value *Pipeline) {
	writeUint64(writer, uint64(c.Lower(value)))
}

func LiftFromExternalPipeline(handle uint64) *Pipeline {
	return FfiConverterPipelineINSTANCE.Lift(C.uint64_t(handle))
}

func LowerToExternalPipeline(value *Pipeline) uint64 {
	return uint64(FfiConverterPipelineINSTANCE.Lower(value))
}

type FfiDestroyerPipeline struct{}

func (_ FfiDestroyerPipeline) Destroy(value *Pipeline) {
	value.Destroy()
}

// Builder for a [`Pipeline`].
//
// Use [`PipelineBuilder::new`] to start, attach workflows via
// [`add_workflow`](Self::add_workflow) / [`stage`](Self::stage) /
// [`parallel`](Self::parallel), then call [`build`](Self::build) to
// validate and produce a runnable [`Pipeline`].
type PipelineBuilderInterface interface {
	// Append a sequential workflow stage with an auto-generated stage name
	// of the form `"stage-{N}"` (zero-based).
	//
	// Use [`stage`](Self::stage) when the stage name matters for
	// downstream tooling that filters by it.
	AddWorkflow(workflow *Workflow) (*PipelineBuilder, error)
	// Validate the pipeline definition and produce a runnable
	// [`Pipeline`].
	//
	// Fails with [`BlazenError::Validation`] if the pipeline has zero
	// stages or if any stage names are duplicated.
	Build() (*Pipeline, error)
	// Append a parallel stage running multiple workflows concurrently.
	//
	// `branch_names` and `workflows` are positionally paired; a length
	// mismatch yields [`BlazenError::Validation`]. When `wait_all` is
	// `true` every branch must complete and outputs are collected into a
	// JSON object keyed by branch name. When `wait_all` is `false` the
	// pipeline proceeds as soon as the first branch finishes and the
	// remaining branches are dropped (which aborts their inner workflows
	// via `WorkflowHandler`'s `Drop` impl).
	Parallel(name string, branchNames []string, workflows []*Workflow, waitAll bool) (*PipelineBuilder, error)
	// Append a sequential stage with an explicit name. The stage name must
	// be unique within the pipeline (enforced at [`build`](Self::build)).
	Stage(name string, workflow *Workflow) (*PipelineBuilder, error)
	// Per-stage timeout in milliseconds. Each stage's workflow gets at
	// most this long to produce its `StopEvent` before the pipeline
	// aborts with [`BlazenError::Timeout`].
	TimeoutPerStageMs(millis uint64) (*PipelineBuilder, error)
	// Total wall-clock timeout for the entire pipeline run, in
	// milliseconds. The pipeline aborts with [`BlazenError::Timeout`] if
	// it does not finish within this duration.
	TotalTimeoutMs(millis uint64) (*PipelineBuilder, error)
}

// Builder for a [`Pipeline`].
//
// Use [`PipelineBuilder::new`] to start, attach workflows via
// [`add_workflow`](Self::add_workflow) / [`stage`](Self::stage) /
// [`parallel`](Self::parallel), then call [`build`](Self::build) to
// validate and produce a runnable [`Pipeline`].
type PipelineBuilder struct {
	ffiObject FfiObject
}

// Create a new builder with the given pipeline name.
func NewPipelineBuilder(name string) *PipelineBuilder {
	return FfiConverterPipelineBuilderINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_constructor_pipelinebuilder_new(FfiConverterStringINSTANCE.Lower(name), _uniffiStatus)
	}))
}

// Append a sequential workflow stage with an auto-generated stage name
// of the form `"stage-{N}"` (zero-based).
//
// Use [`stage`](Self::stage) when the stage name matters for
// downstream tooling that filters by it.
func (_self *PipelineBuilder) AddWorkflow(workflow *Workflow) (*PipelineBuilder, error) {
	_pointer := _self.ffiObject.incrementPointer("*PipelineBuilder")
	defer _self.ffiObject.decrementPointer()
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_method_pipelinebuilder_add_workflow(
			_pointer, FfiConverterWorkflowINSTANCE.Lower(workflow), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *PipelineBuilder
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterPipelineBuilderINSTANCE.Lift(_uniffiRV), nil
	}
}

// Validate the pipeline definition and produce a runnable
// [`Pipeline`].
//
// Fails with [`BlazenError::Validation`] if the pipeline has zero
// stages or if any stage names are duplicated.
func (_self *PipelineBuilder) Build() (*Pipeline, error) {
	_pointer := _self.ffiObject.incrementPointer("*PipelineBuilder")
	defer _self.ffiObject.decrementPointer()
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_method_pipelinebuilder_build(
			_pointer, _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *Pipeline
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterPipelineINSTANCE.Lift(_uniffiRV), nil
	}
}

// Append a parallel stage running multiple workflows concurrently.
//
// `branch_names` and `workflows` are positionally paired; a length
// mismatch yields [`BlazenError::Validation`]. When `wait_all` is
// `true` every branch must complete and outputs are collected into a
// JSON object keyed by branch name. When `wait_all` is `false` the
// pipeline proceeds as soon as the first branch finishes and the
// remaining branches are dropped (which aborts their inner workflows
// via `WorkflowHandler`'s `Drop` impl).
func (_self *PipelineBuilder) Parallel(name string, branchNames []string, workflows []*Workflow, waitAll bool) (*PipelineBuilder, error) {
	_pointer := _self.ffiObject.incrementPointer("*PipelineBuilder")
	defer _self.ffiObject.decrementPointer()
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_method_pipelinebuilder_parallel(
			_pointer, FfiConverterStringINSTANCE.Lower(name), FfiConverterSequenceStringINSTANCE.Lower(branchNames), FfiConverterSequenceWorkflowINSTANCE.Lower(workflows), FfiConverterBoolINSTANCE.Lower(waitAll), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *PipelineBuilder
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterPipelineBuilderINSTANCE.Lift(_uniffiRV), nil
	}
}

// Append a sequential stage with an explicit name. The stage name must
// be unique within the pipeline (enforced at [`build`](Self::build)).
func (_self *PipelineBuilder) Stage(name string, workflow *Workflow) (*PipelineBuilder, error) {
	_pointer := _self.ffiObject.incrementPointer("*PipelineBuilder")
	defer _self.ffiObject.decrementPointer()
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_method_pipelinebuilder_stage(
			_pointer, FfiConverterStringINSTANCE.Lower(name), FfiConverterWorkflowINSTANCE.Lower(workflow), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *PipelineBuilder
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterPipelineBuilderINSTANCE.Lift(_uniffiRV), nil
	}
}

// Per-stage timeout in milliseconds. Each stage's workflow gets at
// most this long to produce its `StopEvent` before the pipeline
// aborts with [`BlazenError::Timeout`].
func (_self *PipelineBuilder) TimeoutPerStageMs(millis uint64) (*PipelineBuilder, error) {
	_pointer := _self.ffiObject.incrementPointer("*PipelineBuilder")
	defer _self.ffiObject.decrementPointer()
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_method_pipelinebuilder_timeout_per_stage_ms(
			_pointer, FfiConverterUint64INSTANCE.Lower(millis), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *PipelineBuilder
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterPipelineBuilderINSTANCE.Lift(_uniffiRV), nil
	}
}

// Total wall-clock timeout for the entire pipeline run, in
// milliseconds. The pipeline aborts with [`BlazenError::Timeout`] if
// it does not finish within this duration.
func (_self *PipelineBuilder) TotalTimeoutMs(millis uint64) (*PipelineBuilder, error) {
	_pointer := _self.ffiObject.incrementPointer("*PipelineBuilder")
	defer _self.ffiObject.decrementPointer()
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_method_pipelinebuilder_total_timeout_ms(
			_pointer, FfiConverterUint64INSTANCE.Lower(millis), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *PipelineBuilder
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterPipelineBuilderINSTANCE.Lift(_uniffiRV), nil
	}
}
func (object *PipelineBuilder) Destroy() {
	runtime.SetFinalizer(object, nil)
	object.ffiObject.destroy()
}

type FfiConverterPipelineBuilder struct{}

var FfiConverterPipelineBuilderINSTANCE = FfiConverterPipelineBuilder{}

func (c FfiConverterPipelineBuilder) Lift(handle C.uint64_t) *PipelineBuilder {
	result := &PipelineBuilder{
		newFfiObject(
			handle,
			func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
				return C.uniffi_blazen_uniffi_fn_clone_pipelinebuilder(handle, status)
			},
			func(handle C.uint64_t, status *C.RustCallStatus) {
				C.uniffi_blazen_uniffi_fn_free_pipelinebuilder(handle, status)
			},
		),
	}
	runtime.SetFinalizer(result, (*PipelineBuilder).Destroy)
	return result
}

func (c FfiConverterPipelineBuilder) Read(reader io.Reader) *PipelineBuilder {
	return c.Lift(C.uint64_t(readUint64(reader)))
}

func (c FfiConverterPipelineBuilder) Lower(value *PipelineBuilder) C.uint64_t {
	// TODO: this is bad - all synchronization from ObjectRuntime.go is discarded here,
	// because the handle will be decremented immediately after this function returns,
	// and someone will be left holding onto a non-locked handle.
	handle := value.ffiObject.incrementPointer("*PipelineBuilder")
	defer value.ffiObject.decrementPointer()
	return handle
}

func (c FfiConverterPipelineBuilder) Write(writer io.Writer, value *PipelineBuilder) {
	writeUint64(writer, uint64(c.Lower(value)))
}

func LiftFromExternalPipelineBuilder(handle uint64) *PipelineBuilder {
	return FfiConverterPipelineBuilderINSTANCE.Lift(C.uint64_t(handle))
}

func LowerToExternalPipelineBuilder(value *PipelineBuilder) uint64 {
	return uint64(FfiConverterPipelineBuilderINSTANCE.Lower(value))
}

type FfiDestroyerPipelineBuilder struct{}

func (_ FfiDestroyerPipelineBuilder) Destroy(value *PipelineBuilder) {
	value.Destroy()
}

// Step handler implemented in foreign code (Go / Swift / Kotlin / Ruby).
//
// The Rust workflow engine calls `invoke` whenever an event matching the
// step's `accepts` list arrives, and routes the returned [`StepOutput`]
// back into the event queue.
//
// ## Async story
//
// `invoke` is `async` on the Rust side. UniFFI exposes this as:
// - Go: blocking function, safe to call from goroutines (composes with channels)
// - Swift: `async throws` method
// - Kotlin: `suspend fun` method
// - Ruby: blocking method (wrap in `Async { ... }` block for fiber concurrency)
type StepHandler interface {
	Invoke(event Event) (StepOutput, error)
}

// Step handler implemented in foreign code (Go / Swift / Kotlin / Ruby).
//
// The Rust workflow engine calls `invoke` whenever an event matching the
// step's `accepts` list arrives, and routes the returned [`StepOutput`]
// back into the event queue.
//
// ## Async story
//
// `invoke` is `async` on the Rust side. UniFFI exposes this as:
// - Go: blocking function, safe to call from goroutines (composes with channels)
// - Swift: `async throws` method
// - Kotlin: `suspend fun` method
// - Ruby: blocking method (wrap in `Async { ... }` block for fiber concurrency)
type StepHandlerImpl struct {
	ffiObject FfiObject
}

func (_self *StepHandlerImpl) Invoke(event Event) (StepOutput, error) {
	_pointer := _self.ffiObject.incrementPointer("StepHandler")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) StepOutput {
			return FfiConverterStepOutputINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_stephandler_invoke(
			_pointer, FfiConverterEventINSTANCE.Lower(event)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}
func (object *StepHandlerImpl) Destroy() {
	runtime.SetFinalizer(object, nil)
	object.ffiObject.destroy()
}

type FfiConverterStepHandler struct {
	handleMap *concurrentHandleMap[StepHandler]
}

var FfiConverterStepHandlerINSTANCE = FfiConverterStepHandler{
	handleMap: newConcurrentHandleMap[StepHandler](),
}

func (c FfiConverterStepHandler) Lift(handle C.uint64_t) StepHandler {
	if uint64(handle)&1 == 0 {
		// Rust-generated handle (even), construct a new object wrapping the handle
		result := &StepHandlerImpl{
			newFfiObject(
				handle,
				func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
					return C.uniffi_blazen_uniffi_fn_clone_stephandler(handle, status)
				},
				func(handle C.uint64_t, status *C.RustCallStatus) {
					C.uniffi_blazen_uniffi_fn_free_stephandler(handle, status)
				},
			),
		}
		runtime.SetFinalizer(result, (*StepHandlerImpl).Destroy)
		return result
	} else {
		// Go-generated handle (odd), retrieve from the handle map
		val, ok := c.handleMap.tryGet(uint64(handle))
		if !ok {
			panic(fmt.Errorf("no callback in handle map: %d", handle))
		}
		c.handleMap.remove(uint64(handle))
		return val
	}
}

func (c FfiConverterStepHandler) Read(reader io.Reader) StepHandler {
	return c.Lift(C.uint64_t(readUint64(reader)))
}

func (c FfiConverterStepHandler) Lower(value StepHandler) C.uint64_t {
	// TODO: this is bad - all synchronization from ObjectRuntime.go is discarded here,
	// because the handle will be decremented immediately after this function returns,
	// and someone will be left holding onto a non-locked handle.
	if val, ok := value.(*StepHandlerImpl); ok {
		// Rust-backed object, clone the handle
		handle := val.ffiObject.incrementPointer("StepHandler")
		defer val.ffiObject.decrementPointer()
		return handle
	} else {
		// Go-backed object, insert into handle map
		return C.uint64_t(c.handleMap.insert(value))
	}
}

func (c FfiConverterStepHandler) Write(writer io.Writer, value StepHandler) {
	writeUint64(writer, uint64(c.Lower(value)))
}

func LiftFromExternalStepHandler(handle uint64) StepHandler {
	return FfiConverterStepHandlerINSTANCE.Lift(C.uint64_t(handle))
}

func LowerToExternalStepHandler(value StepHandler) uint64 {
	return uint64(FfiConverterStepHandlerINSTANCE.Lower(value))
}

type FfiDestroyerStepHandler struct{}

func (_ FfiDestroyerStepHandler) Destroy(value StepHandler) {
	if val, ok := value.(*StepHandlerImpl); ok {
		val.Destroy()
	}
}

//export blazen_uniffi_workflow_cgo_dispatchCallbackInterfaceStepHandlerMethod0
func blazen_uniffi_workflow_cgo_dispatchCallbackInterfaceStepHandlerMethod0(uniffiHandle C.uint64_t, event C.RustBuffer, uniffiFutureCallback C.UniffiForeignFutureCompleteRustBuffer, uniffiCallbackData C.uint64_t, uniffiOutDroppedCallback *C.UniffiForeignFutureDroppedCallbackStruct) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterStepHandlerINSTANCE.handleMap.tryGet(handle)
	if !ok {
		panic(fmt.Errorf("no callback in handle map: %d", handle))
	}

	result := make(chan C.UniffiForeignFutureResultRustBuffer, 1)
	cancel := make(chan struct{}, 1)
	guardHandle := cgo.NewHandle(cancel)
	*uniffiOutDroppedCallback = C.UniffiForeignFutureDroppedCallbackStruct{
		handle: C.uint64_t(guardHandle),
		free:   C.UniffiForeignFutureDroppedCallback(C.blazen_uniffiFreeGorutine),
	}

	// Wait for compleation or cancel
	go func() {
		select {
		case <-cancel:
		case res := <-result:
			C.call_UniffiForeignFutureCompleteRustBuffer(uniffiFutureCallback, uniffiCallbackData, res)
		}
	}()

	// Eval callback asynchroniously
	go func() {
		asyncResult := &C.UniffiForeignFutureResultRustBuffer{}
		uniffiOutReturn := &asyncResult.returnValue
		callStatus := &asyncResult.callStatus
		defer func() {
			result <- *asyncResult
		}()

		res, err :=
			uniffiObj.Invoke(
				FfiConverterEventINSTANCE.Lift(GoRustBuffer{
					inner: event,
				}),
			)

		if err != nil {
			var actualError *BlazenError
			if errors.As(err, &actualError) {
				*callStatus = C.RustCallStatus{
					code:     C.int8_t(uniffiCallbackResultError),
					errorBuf: FfiConverterBlazenErrorINSTANCE.Lower(actualError),
				}
			} else {
				*callStatus = C.RustCallStatus{
					code: C.int8_t(uniffiCallbackUnexpectedResultError),
				}
			}
			return
		}

		*uniffiOutReturn = FfiConverterStepOutputINSTANCE.Lower(res)
	}()
}

var UniffiVTableCallbackInterfaceStepHandlerINSTANCE = C.UniffiVTableCallbackInterfaceStepHandler{
	uniffiFree:  (C.UniffiCallbackInterfaceFree)(C.blazen_uniffi_workflow_cgo_dispatchCallbackInterfaceStepHandlerFree),
	uniffiClone: (C.UniffiCallbackInterfaceClone)(C.blazen_uniffi_workflow_cgo_dispatchCallbackInterfaceStepHandlerClone),
	invoke:      (C.UniffiCallbackInterfaceStepHandlerMethod0)(C.blazen_uniffi_workflow_cgo_dispatchCallbackInterfaceStepHandlerMethod0),
}

//export blazen_uniffi_workflow_cgo_dispatchCallbackInterfaceStepHandlerFree
func blazen_uniffi_workflow_cgo_dispatchCallbackInterfaceStepHandlerFree(handle C.uint64_t) {
	FfiConverterStepHandlerINSTANCE.handleMap.remove(uint64(handle))
}

//export blazen_uniffi_workflow_cgo_dispatchCallbackInterfaceStepHandlerClone
func blazen_uniffi_workflow_cgo_dispatchCallbackInterfaceStepHandlerClone(handle C.uint64_t) C.uint64_t {
	val, ok := FfiConverterStepHandlerINSTANCE.handleMap.tryGet(uint64(handle))
	if !ok {
		panic(fmt.Errorf("no callback in handle map: %d", handle))
	}
	return C.uint64_t(FfiConverterStepHandlerINSTANCE.handleMap.insert(val))
}

func (c FfiConverterStepHandler) register() {
	C.uniffi_blazen_uniffi_fn_init_callback_vtable_stephandler(&UniffiVTableCallbackInterfaceStepHandlerINSTANCE)
}

// A speech-to-text model.
//
// Construct via [`new_whisper_stt_model`] (local, feature-gated) or
// [`new_fal_stt_model`] (cloud). Once obtained, call
// [`transcribe`](Self::transcribe) (async) or
// [`transcribe_blocking`](Self::transcribe_blocking) (sync) to transcribe
// audio.
type SttModelInterface interface {
	// Transcribe audio at `audio_source` and return the transcript.
	//
	// `audio_source` is interpreted per-backend: the whisper.cpp backend
	// treats it as a local file path (16-bit PCM mono WAV at 16 kHz);
	// fal.ai treats it as an HTTP(S) URL or a `data:` URI. `language` is
	// an optional ISO-639-1 hint — when omitted, providers that support
	// language detection will auto-detect.
	Transcribe(audioSource string, language *string) (SttResult, error)
	// Synchronous variant of [`transcribe`](Self::transcribe).
	TranscribeBlocking(audioSource string, language *string) (SttResult, error)
}

// A speech-to-text model.
//
// Construct via [`new_whisper_stt_model`] (local, feature-gated) or
// [`new_fal_stt_model`] (cloud). Once obtained, call
// [`transcribe`](Self::transcribe) (async) or
// [`transcribe_blocking`](Self::transcribe_blocking) (sync) to transcribe
// audio.
type SttModel struct {
	ffiObject FfiObject
}

// Transcribe audio at `audio_source` and return the transcript.
//
// `audio_source` is interpreted per-backend: the whisper.cpp backend
// treats it as a local file path (16-bit PCM mono WAV at 16 kHz);
// fal.ai treats it as an HTTP(S) URL or a `data:` URI. `language` is
// an optional ISO-639-1 hint — when omitted, providers that support
// language detection will auto-detect.
func (_self *SttModel) Transcribe(audioSource string, language *string) (SttResult, error) {
	_pointer := _self.ffiObject.incrementPointer("*SttModel")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) SttResult {
			return FfiConverterSttResultINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_sttmodel_transcribe(
			_pointer, FfiConverterStringINSTANCE.Lower(audioSource), FfiConverterOptionalStringINSTANCE.Lower(language)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// Synchronous variant of [`transcribe`](Self::transcribe).
func (_self *SttModel) TranscribeBlocking(audioSource string, language *string) (SttResult, error) {
	_pointer := _self.ffiObject.incrementPointer("*SttModel")
	defer _self.ffiObject.decrementPointer()
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) RustBufferI {
		return GoRustBuffer{
			inner: C.uniffi_blazen_uniffi_fn_method_sttmodel_transcribe_blocking(
				_pointer, FfiConverterStringINSTANCE.Lower(audioSource), FfiConverterOptionalStringINSTANCE.Lower(language), _uniffiStatus),
		}
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue SttResult
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterSttResultINSTANCE.Lift(_uniffiRV), nil
	}
}
func (object *SttModel) Destroy() {
	runtime.SetFinalizer(object, nil)
	object.ffiObject.destroy()
}

type FfiConverterSttModel struct{}

var FfiConverterSttModelINSTANCE = FfiConverterSttModel{}

func (c FfiConverterSttModel) Lift(handle C.uint64_t) *SttModel {
	result := &SttModel{
		newFfiObject(
			handle,
			func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
				return C.uniffi_blazen_uniffi_fn_clone_sttmodel(handle, status)
			},
			func(handle C.uint64_t, status *C.RustCallStatus) {
				C.uniffi_blazen_uniffi_fn_free_sttmodel(handle, status)
			},
		),
	}
	runtime.SetFinalizer(result, (*SttModel).Destroy)
	return result
}

func (c FfiConverterSttModel) Read(reader io.Reader) *SttModel {
	return c.Lift(C.uint64_t(readUint64(reader)))
}

func (c FfiConverterSttModel) Lower(value *SttModel) C.uint64_t {
	// TODO: this is bad - all synchronization from ObjectRuntime.go is discarded here,
	// because the handle will be decremented immediately after this function returns,
	// and someone will be left holding onto a non-locked handle.
	handle := value.ffiObject.incrementPointer("*SttModel")
	defer value.ffiObject.decrementPointer()
	return handle
}

func (c FfiConverterSttModel) Write(writer io.Writer, value *SttModel) {
	writeUint64(writer, uint64(c.Lower(value)))
}

func LiftFromExternalSttModel(handle uint64) *SttModel {
	return FfiConverterSttModelINSTANCE.Lift(C.uint64_t(handle))
}

func LowerToExternalSttModel(value *SttModel) uint64 {
	return uint64(FfiConverterSttModelINSTANCE.Lower(value))
}

type FfiDestroyerSttModel struct{}

func (_ FfiDestroyerSttModel) Destroy(value *SttModel) {
	value.Destroy()
}

// Foreign-language tool executor invoked by the agent loop.
//
// Implementations receive the LLM's chosen `tool_name` plus a JSON-encoded
// `arguments_json` string and return a JSON-encoded result string that is
// fed back to the model on the next turn.
//
// ## Errors
//
// Returning [`BlazenError`] from `execute` aborts the agent loop with that
// error. Use [`BlazenError::Tool`] for handler-side failures; the message is
// surfaced verbatim to the foreign caller.
type ToolHandler interface {
	// Execute the named tool with JSON-encoded arguments.
	//
	// The returned string is JSON-encoded and round-trips back into the LLM
	// as the tool result on the next turn. Return `"null"` (the JSON literal)
	// when the tool produced no useful result.
	Execute(toolName string, argumentsJson string) (string, error)
}

// Foreign-language tool executor invoked by the agent loop.
//
// Implementations receive the LLM's chosen `tool_name` plus a JSON-encoded
// `arguments_json` string and return a JSON-encoded result string that is
// fed back to the model on the next turn.
//
// ## Errors
//
// Returning [`BlazenError`] from `execute` aborts the agent loop with that
// error. Use [`BlazenError::Tool`] for handler-side failures; the message is
// surfaced verbatim to the foreign caller.
type ToolHandlerImpl struct {
	ffiObject FfiObject
}

// Execute the named tool with JSON-encoded arguments.
//
// The returned string is JSON-encoded and round-trips back into the LLM
// as the tool result on the next turn. Return `"null"` (the JSON literal)
// when the tool produced no useful result.
func (_self *ToolHandlerImpl) Execute(toolName string, argumentsJson string) (string, error) {
	_pointer := _self.ffiObject.incrementPointer("ToolHandler")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) string {
			return FfiConverterStringINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_toolhandler_execute(
			_pointer, FfiConverterStringINSTANCE.Lower(toolName), FfiConverterStringINSTANCE.Lower(argumentsJson)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}
func (object *ToolHandlerImpl) Destroy() {
	runtime.SetFinalizer(object, nil)
	object.ffiObject.destroy()
}

type FfiConverterToolHandler struct {
	handleMap *concurrentHandleMap[ToolHandler]
}

var FfiConverterToolHandlerINSTANCE = FfiConverterToolHandler{
	handleMap: newConcurrentHandleMap[ToolHandler](),
}

func (c FfiConverterToolHandler) Lift(handle C.uint64_t) ToolHandler {
	if uint64(handle)&1 == 0 {
		// Rust-generated handle (even), construct a new object wrapping the handle
		result := &ToolHandlerImpl{
			newFfiObject(
				handle,
				func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
					return C.uniffi_blazen_uniffi_fn_clone_toolhandler(handle, status)
				},
				func(handle C.uint64_t, status *C.RustCallStatus) {
					C.uniffi_blazen_uniffi_fn_free_toolhandler(handle, status)
				},
			),
		}
		runtime.SetFinalizer(result, (*ToolHandlerImpl).Destroy)
		return result
	} else {
		// Go-generated handle (odd), retrieve from the handle map
		val, ok := c.handleMap.tryGet(uint64(handle))
		if !ok {
			panic(fmt.Errorf("no callback in handle map: %d", handle))
		}
		c.handleMap.remove(uint64(handle))
		return val
	}
}

func (c FfiConverterToolHandler) Read(reader io.Reader) ToolHandler {
	return c.Lift(C.uint64_t(readUint64(reader)))
}

func (c FfiConverterToolHandler) Lower(value ToolHandler) C.uint64_t {
	// TODO: this is bad - all synchronization from ObjectRuntime.go is discarded here,
	// because the handle will be decremented immediately after this function returns,
	// and someone will be left holding onto a non-locked handle.
	if val, ok := value.(*ToolHandlerImpl); ok {
		// Rust-backed object, clone the handle
		handle := val.ffiObject.incrementPointer("ToolHandler")
		defer val.ffiObject.decrementPointer()
		return handle
	} else {
		// Go-backed object, insert into handle map
		return C.uint64_t(c.handleMap.insert(value))
	}
}

func (c FfiConverterToolHandler) Write(writer io.Writer, value ToolHandler) {
	writeUint64(writer, uint64(c.Lower(value)))
}

func LiftFromExternalToolHandler(handle uint64) ToolHandler {
	return FfiConverterToolHandlerINSTANCE.Lift(C.uint64_t(handle))
}

func LowerToExternalToolHandler(value ToolHandler) uint64 {
	return uint64(FfiConverterToolHandlerINSTANCE.Lower(value))
}

type FfiDestroyerToolHandler struct{}

func (_ FfiDestroyerToolHandler) Destroy(value ToolHandler) {
	if val, ok := value.(*ToolHandlerImpl); ok {
		val.Destroy()
	}
}

//export blazen_uniffi_agent_cgo_dispatchCallbackInterfaceToolHandlerMethod0
func blazen_uniffi_agent_cgo_dispatchCallbackInterfaceToolHandlerMethod0(uniffiHandle C.uint64_t, toolName C.RustBuffer, argumentsJson C.RustBuffer, uniffiFutureCallback C.UniffiForeignFutureCompleteRustBuffer, uniffiCallbackData C.uint64_t, uniffiOutDroppedCallback *C.UniffiForeignFutureDroppedCallbackStruct) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterToolHandlerINSTANCE.handleMap.tryGet(handle)
	if !ok {
		panic(fmt.Errorf("no callback in handle map: %d", handle))
	}

	result := make(chan C.UniffiForeignFutureResultRustBuffer, 1)
	cancel := make(chan struct{}, 1)
	guardHandle := cgo.NewHandle(cancel)
	*uniffiOutDroppedCallback = C.UniffiForeignFutureDroppedCallbackStruct{
		handle: C.uint64_t(guardHandle),
		free:   C.UniffiForeignFutureDroppedCallback(C.blazen_uniffiFreeGorutine),
	}

	// Wait for compleation or cancel
	go func() {
		select {
		case <-cancel:
		case res := <-result:
			C.call_UniffiForeignFutureCompleteRustBuffer(uniffiFutureCallback, uniffiCallbackData, res)
		}
	}()

	// Eval callback asynchroniously
	go func() {
		asyncResult := &C.UniffiForeignFutureResultRustBuffer{}
		uniffiOutReturn := &asyncResult.returnValue
		callStatus := &asyncResult.callStatus
		defer func() {
			result <- *asyncResult
		}()

		res, err :=
			uniffiObj.Execute(
				FfiConverterStringINSTANCE.Lift(GoRustBuffer{
					inner: toolName,
				}),
				FfiConverterStringINSTANCE.Lift(GoRustBuffer{
					inner: argumentsJson,
				}),
			)

		if err != nil {
			var actualError *BlazenError
			if errors.As(err, &actualError) {
				*callStatus = C.RustCallStatus{
					code:     C.int8_t(uniffiCallbackResultError),
					errorBuf: FfiConverterBlazenErrorINSTANCE.Lower(actualError),
				}
			} else {
				*callStatus = C.RustCallStatus{
					code: C.int8_t(uniffiCallbackUnexpectedResultError),
				}
			}
			return
		}

		*uniffiOutReturn = FfiConverterStringINSTANCE.Lower(res)
	}()
}

var UniffiVTableCallbackInterfaceToolHandlerINSTANCE = C.UniffiVTableCallbackInterfaceToolHandler{
	uniffiFree:  (C.UniffiCallbackInterfaceFree)(C.blazen_uniffi_agent_cgo_dispatchCallbackInterfaceToolHandlerFree),
	uniffiClone: (C.UniffiCallbackInterfaceClone)(C.blazen_uniffi_agent_cgo_dispatchCallbackInterfaceToolHandlerClone),
	execute:     (C.UniffiCallbackInterfaceToolHandlerMethod0)(C.blazen_uniffi_agent_cgo_dispatchCallbackInterfaceToolHandlerMethod0),
}

//export blazen_uniffi_agent_cgo_dispatchCallbackInterfaceToolHandlerFree
func blazen_uniffi_agent_cgo_dispatchCallbackInterfaceToolHandlerFree(handle C.uint64_t) {
	FfiConverterToolHandlerINSTANCE.handleMap.remove(uint64(handle))
}

//export blazen_uniffi_agent_cgo_dispatchCallbackInterfaceToolHandlerClone
func blazen_uniffi_agent_cgo_dispatchCallbackInterfaceToolHandlerClone(handle C.uint64_t) C.uint64_t {
	val, ok := FfiConverterToolHandlerINSTANCE.handleMap.tryGet(uint64(handle))
	if !ok {
		panic(fmt.Errorf("no callback in handle map: %d", handle))
	}
	return C.uint64_t(FfiConverterToolHandlerINSTANCE.handleMap.insert(val))
}

func (c FfiConverterToolHandler) register() {
	C.uniffi_blazen_uniffi_fn_init_callback_vtable_toolhandler(&UniffiVTableCallbackInterfaceToolHandlerINSTANCE)
}

// A text-to-speech model.
//
// Construct via [`new_piper_tts_model`] (local, feature-gated) or
// [`new_fal_tts_model`] (cloud). Once obtained, call
// [`synthesize`](Self::synthesize) (async) or
// [`synthesize_blocking`](Self::synthesize_blocking) (sync) to generate
// speech.
type TtsModelInterface interface {
	// Synthesize speech from `text` and return the audio payload.
	//
	// `voice` selects a provider-specific voice id; `language` is an
	// optional ISO-639-1 hint. Both are ignored by providers that don't
	// support them.
	Synthesize(text string, voice *string, language *string) (TtsResult, error)
	// Synchronous variant of [`synthesize`](Self::synthesize) — blocks on
	// the shared Tokio runtime.
	SynthesizeBlocking(text string, voice *string, language *string) (TtsResult, error)
}

// A text-to-speech model.
//
// Construct via [`new_piper_tts_model`] (local, feature-gated) or
// [`new_fal_tts_model`] (cloud). Once obtained, call
// [`synthesize`](Self::synthesize) (async) or
// [`synthesize_blocking`](Self::synthesize_blocking) (sync) to generate
// speech.
type TtsModel struct {
	ffiObject FfiObject
}

// Synthesize speech from `text` and return the audio payload.
//
// `voice` selects a provider-specific voice id; `language` is an
// optional ISO-639-1 hint. Both are ignored by providers that don't
// support them.
func (_self *TtsModel) Synthesize(text string, voice *string, language *string) (TtsResult, error) {
	_pointer := _self.ffiObject.incrementPointer("*TtsModel")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) TtsResult {
			return FfiConverterTtsResultINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_ttsmodel_synthesize(
			_pointer, FfiConverterStringINSTANCE.Lower(text), FfiConverterOptionalStringINSTANCE.Lower(voice), FfiConverterOptionalStringINSTANCE.Lower(language)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// Synchronous variant of [`synthesize`](Self::synthesize) — blocks on
// the shared Tokio runtime.
func (_self *TtsModel) SynthesizeBlocking(text string, voice *string, language *string) (TtsResult, error) {
	_pointer := _self.ffiObject.incrementPointer("*TtsModel")
	defer _self.ffiObject.decrementPointer()
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) RustBufferI {
		return GoRustBuffer{
			inner: C.uniffi_blazen_uniffi_fn_method_ttsmodel_synthesize_blocking(
				_pointer, FfiConverterStringINSTANCE.Lower(text), FfiConverterOptionalStringINSTANCE.Lower(voice), FfiConverterOptionalStringINSTANCE.Lower(language), _uniffiStatus),
		}
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue TtsResult
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterTtsResultINSTANCE.Lift(_uniffiRV), nil
	}
}
func (object *TtsModel) Destroy() {
	runtime.SetFinalizer(object, nil)
	object.ffiObject.destroy()
}

type FfiConverterTtsModel struct{}

var FfiConverterTtsModelINSTANCE = FfiConverterTtsModel{}

func (c FfiConverterTtsModel) Lift(handle C.uint64_t) *TtsModel {
	result := &TtsModel{
		newFfiObject(
			handle,
			func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
				return C.uniffi_blazen_uniffi_fn_clone_ttsmodel(handle, status)
			},
			func(handle C.uint64_t, status *C.RustCallStatus) {
				C.uniffi_blazen_uniffi_fn_free_ttsmodel(handle, status)
			},
		),
	}
	runtime.SetFinalizer(result, (*TtsModel).Destroy)
	return result
}

func (c FfiConverterTtsModel) Read(reader io.Reader) *TtsModel {
	return c.Lift(C.uint64_t(readUint64(reader)))
}

func (c FfiConverterTtsModel) Lower(value *TtsModel) C.uint64_t {
	// TODO: this is bad - all synchronization from ObjectRuntime.go is discarded here,
	// because the handle will be decremented immediately after this function returns,
	// and someone will be left holding onto a non-locked handle.
	handle := value.ffiObject.incrementPointer("*TtsModel")
	defer value.ffiObject.decrementPointer()
	return handle
}

func (c FfiConverterTtsModel) Write(writer io.Writer, value *TtsModel) {
	writeUint64(writer, uint64(c.Lower(value)))
}

func LiftFromExternalTtsModel(handle uint64) *TtsModel {
	return FfiConverterTtsModelINSTANCE.Lift(C.uint64_t(handle))
}

func LowerToExternalTtsModel(value *TtsModel) uint64 {
	return uint64(FfiConverterTtsModelINSTANCE.Lower(value))
}

type FfiDestroyerTtsModel struct{}

func (_ FfiDestroyerTtsModel) Destroy(value *TtsModel) {
	value.Destroy()
}

// A built workflow ready to run.
type WorkflowInterface interface {
	// Run the workflow to completion with the given JSON input as the
	// `StartEvent` payload. Blocks (in Go) / suspends (in Swift/Kotlin)
	// until the workflow emits its `StopEvent` (or fails).
	Run(inputJson string) (WorkflowResult, error)
	// Synchronous variant of [`run`] — blocks the current thread on the
	// shared Tokio runtime. Provided for callers that want fire-and-forget
	// usage without the host language's async machinery (handy for Ruby
	// scripts and quick Go main fns). Prefer the async [`run`] in long-
	// running services.
	RunBlocking(inputJson string) (WorkflowResult, error)
	// Names of all registered steps, in registration order.
	StepNames() []string
}

// A built workflow ready to run.
type Workflow struct {
	ffiObject FfiObject
}

// Run the workflow to completion with the given JSON input as the
// `StartEvent` payload. Blocks (in Go) / suspends (in Swift/Kotlin)
// until the workflow emits its `StopEvent` (or fails).
func (_self *Workflow) Run(inputJson string) (WorkflowResult, error) {
	_pointer := _self.ffiObject.incrementPointer("*Workflow")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) WorkflowResult {
			return FfiConverterWorkflowResultINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_workflow_run(
			_pointer, FfiConverterStringINSTANCE.Lower(inputJson)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// Synchronous variant of [`run`] — blocks the current thread on the
// shared Tokio runtime. Provided for callers that want fire-and-forget
// usage without the host language's async machinery (handy for Ruby
// scripts and quick Go main fns). Prefer the async [`run`] in long-
// running services.
func (_self *Workflow) RunBlocking(inputJson string) (WorkflowResult, error) {
	_pointer := _self.ffiObject.incrementPointer("*Workflow")
	defer _self.ffiObject.decrementPointer()
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) RustBufferI {
		return GoRustBuffer{
			inner: C.uniffi_blazen_uniffi_fn_method_workflow_run_blocking(
				_pointer, FfiConverterStringINSTANCE.Lower(inputJson), _uniffiStatus),
		}
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue WorkflowResult
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterWorkflowResultINSTANCE.Lift(_uniffiRV), nil
	}
}

// Names of all registered steps, in registration order.
func (_self *Workflow) StepNames() []string {
	_pointer := _self.ffiObject.incrementPointer("*Workflow")
	defer _self.ffiObject.decrementPointer()
	return FfiConverterSequenceStringINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) RustBufferI {
		return GoRustBuffer{
			inner: C.uniffi_blazen_uniffi_fn_method_workflow_step_names(
				_pointer, _uniffiStatus),
		}
	}))
}
func (object *Workflow) Destroy() {
	runtime.SetFinalizer(object, nil)
	object.ffiObject.destroy()
}

type FfiConverterWorkflow struct{}

var FfiConverterWorkflowINSTANCE = FfiConverterWorkflow{}

func (c FfiConverterWorkflow) Lift(handle C.uint64_t) *Workflow {
	result := &Workflow{
		newFfiObject(
			handle,
			func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
				return C.uniffi_blazen_uniffi_fn_clone_workflow(handle, status)
			},
			func(handle C.uint64_t, status *C.RustCallStatus) {
				C.uniffi_blazen_uniffi_fn_free_workflow(handle, status)
			},
		),
	}
	runtime.SetFinalizer(result, (*Workflow).Destroy)
	return result
}

func (c FfiConverterWorkflow) Read(reader io.Reader) *Workflow {
	return c.Lift(C.uint64_t(readUint64(reader)))
}

func (c FfiConverterWorkflow) Lower(value *Workflow) C.uint64_t {
	// TODO: this is bad - all synchronization from ObjectRuntime.go is discarded here,
	// because the handle will be decremented immediately after this function returns,
	// and someone will be left holding onto a non-locked handle.
	handle := value.ffiObject.incrementPointer("*Workflow")
	defer value.ffiObject.decrementPointer()
	return handle
}

func (c FfiConverterWorkflow) Write(writer io.Writer, value *Workflow) {
	writeUint64(writer, uint64(c.Lower(value)))
}

func LiftFromExternalWorkflow(handle uint64) *Workflow {
	return FfiConverterWorkflowINSTANCE.Lift(C.uint64_t(handle))
}

func LowerToExternalWorkflow(value *Workflow) uint64 {
	return uint64(FfiConverterWorkflowINSTANCE.Lower(value))
}

type FfiDestroyerWorkflow struct{}

func (_ FfiDestroyerWorkflow) Destroy(value *Workflow) {
	value.Destroy()
}

// Builder for [`Workflow`]. Use [`Workflow::builder`] or
// `WorkflowBuilder::new()` to start.
type WorkflowBuilderInterface interface {
	// Consume the builder and produce a [`Workflow`] ready to run.
	Build() (*Workflow, error)
	// Register a step.
	//
	// - `name`: step identifier, must be unique within the workflow.
	// - `accepts`: event-type names this step should be invoked for
	// (e.g. `["StartEvent"]`).
	// - `emits`: event-type names this step is expected to produce. Used for
	// workflow validation and routing; provide every type the handler can
	// return.
	// - `handler`: the foreign-implemented step handler.
	Step(name string, accepts []string, emits []string, handler StepHandler) (*WorkflowBuilder, error)
	// Per-step timeout in milliseconds. Steps that exceed this are aborted.
	StepTimeoutMs(millis uint64) (*WorkflowBuilder, error)
	// Workflow-wide timeout in milliseconds. Whole run aborts after this.
	TimeoutMs(millis uint64) (*WorkflowBuilder, error)
}

// Builder for [`Workflow`]. Use [`Workflow::builder`] or
// `WorkflowBuilder::new()` to start.
type WorkflowBuilder struct {
	ffiObject FfiObject
}

// Create a new builder with the given workflow name.
func NewWorkflowBuilder(name string) *WorkflowBuilder {
	return FfiConverterWorkflowBuilderINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_constructor_workflowbuilder_new(FfiConverterStringINSTANCE.Lower(name), _uniffiStatus)
	}))
}

// Consume the builder and produce a [`Workflow`] ready to run.
func (_self *WorkflowBuilder) Build() (*Workflow, error) {
	_pointer := _self.ffiObject.incrementPointer("*WorkflowBuilder")
	defer _self.ffiObject.decrementPointer()
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_method_workflowbuilder_build(
			_pointer, _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *Workflow
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterWorkflowINSTANCE.Lift(_uniffiRV), nil
	}
}

// Register a step.
//
// - `name`: step identifier, must be unique within the workflow.
// - `accepts`: event-type names this step should be invoked for
// (e.g. `["StartEvent"]`).
// - `emits`: event-type names this step is expected to produce. Used for
// workflow validation and routing; provide every type the handler can
// return.
// - `handler`: the foreign-implemented step handler.
func (_self *WorkflowBuilder) Step(name string, accepts []string, emits []string, handler StepHandler) (*WorkflowBuilder, error) {
	_pointer := _self.ffiObject.incrementPointer("*WorkflowBuilder")
	defer _self.ffiObject.decrementPointer()
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_method_workflowbuilder_step(
			_pointer, FfiConverterStringINSTANCE.Lower(name), FfiConverterSequenceStringINSTANCE.Lower(accepts), FfiConverterSequenceStringINSTANCE.Lower(emits), FfiConverterStepHandlerINSTANCE.Lower(handler), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *WorkflowBuilder
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterWorkflowBuilderINSTANCE.Lift(_uniffiRV), nil
	}
}

// Per-step timeout in milliseconds. Steps that exceed this are aborted.
func (_self *WorkflowBuilder) StepTimeoutMs(millis uint64) (*WorkflowBuilder, error) {
	_pointer := _self.ffiObject.incrementPointer("*WorkflowBuilder")
	defer _self.ffiObject.decrementPointer()
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_method_workflowbuilder_step_timeout_ms(
			_pointer, FfiConverterUint64INSTANCE.Lower(millis), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *WorkflowBuilder
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterWorkflowBuilderINSTANCE.Lift(_uniffiRV), nil
	}
}

// Workflow-wide timeout in milliseconds. Whole run aborts after this.
func (_self *WorkflowBuilder) TimeoutMs(millis uint64) (*WorkflowBuilder, error) {
	_pointer := _self.ffiObject.incrementPointer("*WorkflowBuilder")
	defer _self.ffiObject.decrementPointer()
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_method_workflowbuilder_timeout_ms(
			_pointer, FfiConverterUint64INSTANCE.Lower(millis), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *WorkflowBuilder
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterWorkflowBuilderINSTANCE.Lift(_uniffiRV), nil
	}
}
func (object *WorkflowBuilder) Destroy() {
	runtime.SetFinalizer(object, nil)
	object.ffiObject.destroy()
}

type FfiConverterWorkflowBuilder struct{}

var FfiConverterWorkflowBuilderINSTANCE = FfiConverterWorkflowBuilder{}

func (c FfiConverterWorkflowBuilder) Lift(handle C.uint64_t) *WorkflowBuilder {
	result := &WorkflowBuilder{
		newFfiObject(
			handle,
			func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
				return C.uniffi_blazen_uniffi_fn_clone_workflowbuilder(handle, status)
			},
			func(handle C.uint64_t, status *C.RustCallStatus) {
				C.uniffi_blazen_uniffi_fn_free_workflowbuilder(handle, status)
			},
		),
	}
	runtime.SetFinalizer(result, (*WorkflowBuilder).Destroy)
	return result
}

func (c FfiConverterWorkflowBuilder) Read(reader io.Reader) *WorkflowBuilder {
	return c.Lift(C.uint64_t(readUint64(reader)))
}

func (c FfiConverterWorkflowBuilder) Lower(value *WorkflowBuilder) C.uint64_t {
	// TODO: this is bad - all synchronization from ObjectRuntime.go is discarded here,
	// because the handle will be decremented immediately after this function returns,
	// and someone will be left holding onto a non-locked handle.
	handle := value.ffiObject.incrementPointer("*WorkflowBuilder")
	defer value.ffiObject.decrementPointer()
	return handle
}

func (c FfiConverterWorkflowBuilder) Write(writer io.Writer, value *WorkflowBuilder) {
	writeUint64(writer, uint64(c.Lower(value)))
}

func LiftFromExternalWorkflowBuilder(handle uint64) *WorkflowBuilder {
	return FfiConverterWorkflowBuilderINSTANCE.Lift(C.uint64_t(handle))
}

func LowerToExternalWorkflowBuilder(value *WorkflowBuilder) uint64 {
	return uint64(FfiConverterWorkflowBuilderINSTANCE.Lower(value))
}

type FfiDestroyerWorkflowBuilder struct{}

func (_ FfiDestroyerWorkflowBuilder) Destroy(value *WorkflowBuilder) {
	value.Destroy()
}

// Outcome of an [`Agent::run`] call.
//
// `total_cost_usd` is the sum of per-iteration costs; when the provider did
// not report cost data it is `0.0` (the wire format does not distinguish
// "zero" from "unknown" — foreign callers wanting fidelity should pull
// pricing from telemetry).
type AgentResult struct {
	// The model's final textual response after the loop terminates.
	FinalMessage string
	// Number of iterations (LLM round-trips) the loop executed before
	// terminating.
	Iterations uint32
	// Total number of tool calls executed across all iterations.
	ToolCallCount uint32
	// Aggregated token usage across every completion call in the loop.
	TotalUsage TokenUsage
	// Aggregated USD cost across every completion call in the loop.
	TotalCostUsd float64
}

func (r *AgentResult) Destroy() {
	FfiDestroyerString{}.Destroy(r.FinalMessage)
	FfiDestroyerUint32{}.Destroy(r.Iterations)
	FfiDestroyerUint32{}.Destroy(r.ToolCallCount)
	FfiDestroyerTokenUsage{}.Destroy(r.TotalUsage)
	FfiDestroyerFloat64{}.Destroy(r.TotalCostUsd)
}

type FfiConverterAgentResult struct{}

var FfiConverterAgentResultINSTANCE = FfiConverterAgentResult{}

func (c FfiConverterAgentResult) Lift(rb RustBufferI) AgentResult {
	return LiftFromRustBuffer[AgentResult](c, rb)
}

func (c FfiConverterAgentResult) Read(reader io.Reader) AgentResult {
	return AgentResult{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterUint32INSTANCE.Read(reader),
		FfiConverterUint32INSTANCE.Read(reader),
		FfiConverterTokenUsageINSTANCE.Read(reader),
		FfiConverterFloat64INSTANCE.Read(reader),
	}
}

func (c FfiConverterAgentResult) Lower(value AgentResult) C.RustBuffer {
	return LowerIntoRustBuffer[AgentResult](c, value)
}

func (c FfiConverterAgentResult) LowerExternal(value AgentResult) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[AgentResult](c, value))
}

func (c FfiConverterAgentResult) Write(writer io.Writer, value AgentResult) {
	FfiConverterStringINSTANCE.Write(writer, value.FinalMessage)
	FfiConverterUint32INSTANCE.Write(writer, value.Iterations)
	FfiConverterUint32INSTANCE.Write(writer, value.ToolCallCount)
	FfiConverterTokenUsageINSTANCE.Write(writer, value.TotalUsage)
	FfiConverterFloat64INSTANCE.Write(writer, value.TotalCostUsd)
}

type FfiDestroyerAgentResult struct{}

func (_ FfiDestroyerAgentResult) Destroy(value AgentResult) {
	value.Destroy()
}

type AudioMusicProviderDefaults struct {
	Base *BaseProviderDefaults
}

func (r *AudioMusicProviderDefaults) Destroy() {
	FfiDestroyerOptionalBaseProviderDefaults{}.Destroy(r.Base)
}

type FfiConverterAudioMusicProviderDefaults struct{}

var FfiConverterAudioMusicProviderDefaultsINSTANCE = FfiConverterAudioMusicProviderDefaults{}

func (c FfiConverterAudioMusicProviderDefaults) Lift(rb RustBufferI) AudioMusicProviderDefaults {
	return LiftFromRustBuffer[AudioMusicProviderDefaults](c, rb)
}

func (c FfiConverterAudioMusicProviderDefaults) Read(reader io.Reader) AudioMusicProviderDefaults {
	return AudioMusicProviderDefaults{
		FfiConverterOptionalBaseProviderDefaultsINSTANCE.Read(reader),
	}
}

func (c FfiConverterAudioMusicProviderDefaults) Lower(value AudioMusicProviderDefaults) C.RustBuffer {
	return LowerIntoRustBuffer[AudioMusicProviderDefaults](c, value)
}

func (c FfiConverterAudioMusicProviderDefaults) LowerExternal(value AudioMusicProviderDefaults) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[AudioMusicProviderDefaults](c, value))
}

func (c FfiConverterAudioMusicProviderDefaults) Write(writer io.Writer, value AudioMusicProviderDefaults) {
	FfiConverterOptionalBaseProviderDefaultsINSTANCE.Write(writer, value.Base)
}

type FfiDestroyerAudioMusicProviderDefaults struct{}

func (_ FfiDestroyerAudioMusicProviderDefaults) Destroy(value AudioMusicProviderDefaults) {
	value.Destroy()
}

// Result of an audio generation or TTS operation.
type AudioResult struct {
	Audio        []GeneratedAudio
	Timing       RequestTiming
	Cost         *float64
	Usage        *TokenUsage
	AudioSeconds float64
	Metadata     string
}

func (r *AudioResult) Destroy() {
	FfiDestroyerSequenceGeneratedAudio{}.Destroy(r.Audio)
	FfiDestroyerRequestTiming{}.Destroy(r.Timing)
	FfiDestroyerOptionalFloat64{}.Destroy(r.Cost)
	FfiDestroyerOptionalTokenUsage{}.Destroy(r.Usage)
	FfiDestroyerFloat64{}.Destroy(r.AudioSeconds)
	FfiDestroyerString{}.Destroy(r.Metadata)
}

type FfiConverterAudioResult struct{}

var FfiConverterAudioResultINSTANCE = FfiConverterAudioResult{}

func (c FfiConverterAudioResult) Lift(rb RustBufferI) AudioResult {
	return LiftFromRustBuffer[AudioResult](c, rb)
}

func (c FfiConverterAudioResult) Read(reader io.Reader) AudioResult {
	return AudioResult{
		FfiConverterSequenceGeneratedAudioINSTANCE.Read(reader),
		FfiConverterRequestTimingINSTANCE.Read(reader),
		FfiConverterOptionalFloat64INSTANCE.Read(reader),
		FfiConverterOptionalTokenUsageINSTANCE.Read(reader),
		FfiConverterFloat64INSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
	}
}

func (c FfiConverterAudioResult) Lower(value AudioResult) C.RustBuffer {
	return LowerIntoRustBuffer[AudioResult](c, value)
}

func (c FfiConverterAudioResult) LowerExternal(value AudioResult) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[AudioResult](c, value))
}

func (c FfiConverterAudioResult) Write(writer io.Writer, value AudioResult) {
	FfiConverterSequenceGeneratedAudioINSTANCE.Write(writer, value.Audio)
	FfiConverterRequestTimingINSTANCE.Write(writer, value.Timing)
	FfiConverterOptionalFloat64INSTANCE.Write(writer, value.Cost)
	FfiConverterOptionalTokenUsageINSTANCE.Write(writer, value.Usage)
	FfiConverterFloat64INSTANCE.Write(writer, value.AudioSeconds)
	FfiConverterStringINSTANCE.Write(writer, value.Metadata)
}

type FfiDestroyerAudioResult struct{}

func (_ FfiDestroyerAudioResult) Destroy(value AudioResult) {
	value.Destroy()
}

type AudioSpeechProviderDefaults struct {
	Base *BaseProviderDefaults
}

func (r *AudioSpeechProviderDefaults) Destroy() {
	FfiDestroyerOptionalBaseProviderDefaults{}.Destroy(r.Base)
}

type FfiConverterAudioSpeechProviderDefaults struct{}

var FfiConverterAudioSpeechProviderDefaultsINSTANCE = FfiConverterAudioSpeechProviderDefaults{}

func (c FfiConverterAudioSpeechProviderDefaults) Lift(rb RustBufferI) AudioSpeechProviderDefaults {
	return LiftFromRustBuffer[AudioSpeechProviderDefaults](c, rb)
}

func (c FfiConverterAudioSpeechProviderDefaults) Read(reader io.Reader) AudioSpeechProviderDefaults {
	return AudioSpeechProviderDefaults{
		FfiConverterOptionalBaseProviderDefaultsINSTANCE.Read(reader),
	}
}

func (c FfiConverterAudioSpeechProviderDefaults) Lower(value AudioSpeechProviderDefaults) C.RustBuffer {
	return LowerIntoRustBuffer[AudioSpeechProviderDefaults](c, value)
}

func (c FfiConverterAudioSpeechProviderDefaults) LowerExternal(value AudioSpeechProviderDefaults) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[AudioSpeechProviderDefaults](c, value))
}

func (c FfiConverterAudioSpeechProviderDefaults) Write(writer io.Writer, value AudioSpeechProviderDefaults) {
	FfiConverterOptionalBaseProviderDefaultsINSTANCE.Write(writer, value.Base)
}

type FfiDestroyerAudioSpeechProviderDefaults struct{}

func (_ FfiDestroyerAudioSpeechProviderDefaults) Destroy(value AudioSpeechProviderDefaults) {
	value.Destroy()
}

type BackgroundRemovalProviderDefaults struct {
	Base *BaseProviderDefaults
}

func (r *BackgroundRemovalProviderDefaults) Destroy() {
	FfiDestroyerOptionalBaseProviderDefaults{}.Destroy(r.Base)
}

type FfiConverterBackgroundRemovalProviderDefaults struct{}

var FfiConverterBackgroundRemovalProviderDefaultsINSTANCE = FfiConverterBackgroundRemovalProviderDefaults{}

func (c FfiConverterBackgroundRemovalProviderDefaults) Lift(rb RustBufferI) BackgroundRemovalProviderDefaults {
	return LiftFromRustBuffer[BackgroundRemovalProviderDefaults](c, rb)
}

func (c FfiConverterBackgroundRemovalProviderDefaults) Read(reader io.Reader) BackgroundRemovalProviderDefaults {
	return BackgroundRemovalProviderDefaults{
		FfiConverterOptionalBaseProviderDefaultsINSTANCE.Read(reader),
	}
}

func (c FfiConverterBackgroundRemovalProviderDefaults) Lower(value BackgroundRemovalProviderDefaults) C.RustBuffer {
	return LowerIntoRustBuffer[BackgroundRemovalProviderDefaults](c, value)
}

func (c FfiConverterBackgroundRemovalProviderDefaults) LowerExternal(value BackgroundRemovalProviderDefaults) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[BackgroundRemovalProviderDefaults](c, value))
}

func (c FfiConverterBackgroundRemovalProviderDefaults) Write(writer io.Writer, value BackgroundRemovalProviderDefaults) {
	FfiConverterOptionalBaseProviderDefaultsINSTANCE.Write(writer, value.Base)
}

type FfiDestroyerBackgroundRemovalProviderDefaults struct{}

func (_ FfiDestroyerBackgroundRemovalProviderDefaults) Destroy(value BackgroundRemovalProviderDefaults) {
	value.Destroy()
}

// Request for background removal on an existing image.
type BackgroundRemovalRequest struct {
	ImageUrl   string
	Model      *string
	Parameters string
}

func (r *BackgroundRemovalRequest) Destroy() {
	FfiDestroyerString{}.Destroy(r.ImageUrl)
	FfiDestroyerOptionalString{}.Destroy(r.Model)
	FfiDestroyerString{}.Destroy(r.Parameters)
}

type FfiConverterBackgroundRemovalRequest struct{}

var FfiConverterBackgroundRemovalRequestINSTANCE = FfiConverterBackgroundRemovalRequest{}

func (c FfiConverterBackgroundRemovalRequest) Lift(rb RustBufferI) BackgroundRemovalRequest {
	return LiftFromRustBuffer[BackgroundRemovalRequest](c, rb)
}

func (c FfiConverterBackgroundRemovalRequest) Read(reader io.Reader) BackgroundRemovalRequest {
	return BackgroundRemovalRequest{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
	}
}

func (c FfiConverterBackgroundRemovalRequest) Lower(value BackgroundRemovalRequest) C.RustBuffer {
	return LowerIntoRustBuffer[BackgroundRemovalRequest](c, value)
}

func (c FfiConverterBackgroundRemovalRequest) LowerExternal(value BackgroundRemovalRequest) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[BackgroundRemovalRequest](c, value))
}

func (c FfiConverterBackgroundRemovalRequest) Write(writer io.Writer, value BackgroundRemovalRequest) {
	FfiConverterStringINSTANCE.Write(writer, value.ImageUrl)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.Model)
	FfiConverterStringINSTANCE.Write(writer, value.Parameters)
}

type FfiDestroyerBackgroundRemovalRequest struct{}

func (_ FfiDestroyerBackgroundRemovalRequest) Destroy(value BackgroundRemovalRequest) {
	value.Destroy()
}

// Universal provider defaults applicable across every provider role.
//
// V1 carries no data fields — the upstream `before_request` hook is
// deferred to Phase C. A placeholder boolean field is included so the
// generated foreign-language struct is non-empty (UniFFI Records with
// zero fields generate slightly awkward foreign-side code).
type BaseProviderDefaults struct {
	// Reserved for future use. Currently ignored on both sides of the FFI.
	// V1 carries no universal defaults data — the upstream `before_request`
	// hook is exposed via Phase C's foreign-implementable callback trait.
	Reserved bool
}

func (r *BaseProviderDefaults) Destroy() {
	FfiDestroyerBool{}.Destroy(r.Reserved)
}

type FfiConverterBaseProviderDefaults struct{}

var FfiConverterBaseProviderDefaultsINSTANCE = FfiConverterBaseProviderDefaults{}

func (c FfiConverterBaseProviderDefaults) Lift(rb RustBufferI) BaseProviderDefaults {
	return LiftFromRustBuffer[BaseProviderDefaults](c, rb)
}

func (c FfiConverterBaseProviderDefaults) Read(reader io.Reader) BaseProviderDefaults {
	return BaseProviderDefaults{
		FfiConverterBoolINSTANCE.Read(reader),
	}
}

func (c FfiConverterBaseProviderDefaults) Lower(value BaseProviderDefaults) C.RustBuffer {
	return LowerIntoRustBuffer[BaseProviderDefaults](c, value)
}

func (c FfiConverterBaseProviderDefaults) LowerExternal(value BaseProviderDefaults) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[BaseProviderDefaults](c, value))
}

func (c FfiConverterBaseProviderDefaults) Write(writer io.Writer, value BaseProviderDefaults) {
	FfiConverterBoolINSTANCE.Write(writer, value.Reserved)
}

type FfiDestroyerBaseProviderDefaults struct{}

func (_ FfiDestroyerBaseProviderDefaults) Destroy(value BaseProviderDefaults) {
	value.Destroy()
}

// Outcome of a [`complete_batch`] call.
//
// `total_usage` and `total_cost_usd` aggregate only the successful responses
// — failed slots contribute zero. When no provider reports cost data the
// total is `0.0` (the wire format does not distinguish "zero" from "unknown").
type BatchResult struct {
	// One slot per input request, in the same order.
	Responses []BatchItem
	// Aggregated token usage across successful responses.
	TotalUsage TokenUsage
	// Aggregated USD cost across successful responses.
	TotalCostUsd float64
}

func (r *BatchResult) Destroy() {
	FfiDestroyerSequenceBatchItem{}.Destroy(r.Responses)
	FfiDestroyerTokenUsage{}.Destroy(r.TotalUsage)
	FfiDestroyerFloat64{}.Destroy(r.TotalCostUsd)
}

type FfiConverterBatchResult struct{}

var FfiConverterBatchResultINSTANCE = FfiConverterBatchResult{}

func (c FfiConverterBatchResult) Lift(rb RustBufferI) BatchResult {
	return LiftFromRustBuffer[BatchResult](c, rb)
}

func (c FfiConverterBatchResult) Read(reader io.Reader) BatchResult {
	return BatchResult{
		FfiConverterSequenceBatchItemINSTANCE.Read(reader),
		FfiConverterTokenUsageINSTANCE.Read(reader),
		FfiConverterFloat64INSTANCE.Read(reader),
	}
}

func (c FfiConverterBatchResult) Lower(value BatchResult) C.RustBuffer {
	return LowerIntoRustBuffer[BatchResult](c, value)
}

func (c FfiConverterBatchResult) LowerExternal(value BatchResult) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[BatchResult](c, value))
}

func (c FfiConverterBatchResult) Write(writer io.Writer, value BatchResult) {
	FfiConverterSequenceBatchItemINSTANCE.Write(writer, value.Responses)
	FfiConverterTokenUsageINSTANCE.Write(writer, value.TotalUsage)
	FfiConverterFloat64INSTANCE.Write(writer, value.TotalCostUsd)
}

type FfiDestroyerBatchResult struct{}

func (_ FfiDestroyerBatchResult) Destroy(value BatchResult) {
	value.Destroy()
}

// A single message in a chat conversation.
//
// `role` is one of `"system"`, `"user"`, `"assistant"`, `"tool"`.
// `content` is the text payload (empty string when the message carries only
// tool calls or media). Multimodal media attaches via [`media_parts`].
type ChatMessage struct {
	Role       string
	Content    string
	MediaParts []Media
	ToolCalls  []ToolCall
	ToolCallId *string
	Name       *string
}

func (r *ChatMessage) Destroy() {
	FfiDestroyerString{}.Destroy(r.Role)
	FfiDestroyerString{}.Destroy(r.Content)
	FfiDestroyerSequenceMedia{}.Destroy(r.MediaParts)
	FfiDestroyerSequenceToolCall{}.Destroy(r.ToolCalls)
	FfiDestroyerOptionalString{}.Destroy(r.ToolCallId)
	FfiDestroyerOptionalString{}.Destroy(r.Name)
}

type FfiConverterChatMessage struct{}

var FfiConverterChatMessageINSTANCE = FfiConverterChatMessage{}

func (c FfiConverterChatMessage) Lift(rb RustBufferI) ChatMessage {
	return LiftFromRustBuffer[ChatMessage](c, rb)
}

func (c FfiConverterChatMessage) Read(reader io.Reader) ChatMessage {
	return ChatMessage{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterSequenceMediaINSTANCE.Read(reader),
		FfiConverterSequenceToolCallINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
	}
}

func (c FfiConverterChatMessage) Lower(value ChatMessage) C.RustBuffer {
	return LowerIntoRustBuffer[ChatMessage](c, value)
}

func (c FfiConverterChatMessage) LowerExternal(value ChatMessage) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[ChatMessage](c, value))
}

func (c FfiConverterChatMessage) Write(writer io.Writer, value ChatMessage) {
	FfiConverterStringINSTANCE.Write(writer, value.Role)
	FfiConverterStringINSTANCE.Write(writer, value.Content)
	FfiConverterSequenceMediaINSTANCE.Write(writer, value.MediaParts)
	FfiConverterSequenceToolCallINSTANCE.Write(writer, value.ToolCalls)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.ToolCallId)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.Name)
}

type FfiDestroyerChatMessage struct{}

func (_ FfiDestroyerChatMessage) Destroy(value ChatMessage) {
	value.Destroy()
}

// Completion-role defaults: system prompt, default tools, default
// `response_format`. Hooks (`before_completion`) deferred to Phase C.
type CompletionProviderDefaults struct {
	Base *BaseProviderDefaults
	// Prepended as a system message if the request lacks one.
	SystemPrompt *string
	// JSON-encoded `Vec<ToolDefinition>`. Merged into the request's tool
	// list — request-supplied tools win on name collision.
	ToolsJson *string
	// JSON-encoded `serde_json::Value` for the OpenAI-style
	// `response_format` field. Set only if the request lacks one.
	ResponseFormatJson *string
}

func (r *CompletionProviderDefaults) Destroy() {
	FfiDestroyerOptionalBaseProviderDefaults{}.Destroy(r.Base)
	FfiDestroyerOptionalString{}.Destroy(r.SystemPrompt)
	FfiDestroyerOptionalString{}.Destroy(r.ToolsJson)
	FfiDestroyerOptionalString{}.Destroy(r.ResponseFormatJson)
}

type FfiConverterCompletionProviderDefaults struct{}

var FfiConverterCompletionProviderDefaultsINSTANCE = FfiConverterCompletionProviderDefaults{}

func (c FfiConverterCompletionProviderDefaults) Lift(rb RustBufferI) CompletionProviderDefaults {
	return LiftFromRustBuffer[CompletionProviderDefaults](c, rb)
}

func (c FfiConverterCompletionProviderDefaults) Read(reader io.Reader) CompletionProviderDefaults {
	return CompletionProviderDefaults{
		FfiConverterOptionalBaseProviderDefaultsINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
	}
}

func (c FfiConverterCompletionProviderDefaults) Lower(value CompletionProviderDefaults) C.RustBuffer {
	return LowerIntoRustBuffer[CompletionProviderDefaults](c, value)
}

func (c FfiConverterCompletionProviderDefaults) LowerExternal(value CompletionProviderDefaults) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[CompletionProviderDefaults](c, value))
}

func (c FfiConverterCompletionProviderDefaults) Write(writer io.Writer, value CompletionProviderDefaults) {
	FfiConverterOptionalBaseProviderDefaultsINSTANCE.Write(writer, value.Base)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.SystemPrompt)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.ToolsJson)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.ResponseFormatJson)
}

type FfiDestroyerCompletionProviderDefaults struct{}

func (_ FfiDestroyerCompletionProviderDefaults) Destroy(value CompletionProviderDefaults) {
	value.Destroy()
}

// A provider-agnostic chat completion request.
//
// `system`, when set, is prepended as a `Role::System` message — equivalent
// to building the message list with a leading system entry. Provided as a
// convenience because most foreign callers think of the system prompt as a
// request-level field, not a message.
type CompletionRequest struct {
	Messages    []ChatMessage
	Tools       []Tool
	Temperature *float64
	MaxTokens   *uint32
	TopP        *float64
	// Optional model identifier to use for this request, overriding the
	// provider's default model.
	Model *string
	// Optional JSON Schema (encoded as a string) constraining the model's
	// output. Passed through to upstream's `response_format` slot.
	ResponseFormatJson *string
	// Optional system prompt, prepended as a `system`-role message.
	System *string
}

func (r *CompletionRequest) Destroy() {
	FfiDestroyerSequenceChatMessage{}.Destroy(r.Messages)
	FfiDestroyerSequenceTool{}.Destroy(r.Tools)
	FfiDestroyerOptionalFloat64{}.Destroy(r.Temperature)
	FfiDestroyerOptionalUint32{}.Destroy(r.MaxTokens)
	FfiDestroyerOptionalFloat64{}.Destroy(r.TopP)
	FfiDestroyerOptionalString{}.Destroy(r.Model)
	FfiDestroyerOptionalString{}.Destroy(r.ResponseFormatJson)
	FfiDestroyerOptionalString{}.Destroy(r.System)
}

type FfiConverterCompletionRequest struct{}

var FfiConverterCompletionRequestINSTANCE = FfiConverterCompletionRequest{}

func (c FfiConverterCompletionRequest) Lift(rb RustBufferI) CompletionRequest {
	return LiftFromRustBuffer[CompletionRequest](c, rb)
}

func (c FfiConverterCompletionRequest) Read(reader io.Reader) CompletionRequest {
	return CompletionRequest{
		FfiConverterSequenceChatMessageINSTANCE.Read(reader),
		FfiConverterSequenceToolINSTANCE.Read(reader),
		FfiConverterOptionalFloat64INSTANCE.Read(reader),
		FfiConverterOptionalUint32INSTANCE.Read(reader),
		FfiConverterOptionalFloat64INSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
	}
}

func (c FfiConverterCompletionRequest) Lower(value CompletionRequest) C.RustBuffer {
	return LowerIntoRustBuffer[CompletionRequest](c, value)
}

func (c FfiConverterCompletionRequest) LowerExternal(value CompletionRequest) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[CompletionRequest](c, value))
}

func (c FfiConverterCompletionRequest) Write(writer io.Writer, value CompletionRequest) {
	FfiConverterSequenceChatMessageINSTANCE.Write(writer, value.Messages)
	FfiConverterSequenceToolINSTANCE.Write(writer, value.Tools)
	FfiConverterOptionalFloat64INSTANCE.Write(writer, value.Temperature)
	FfiConverterOptionalUint32INSTANCE.Write(writer, value.MaxTokens)
	FfiConverterOptionalFloat64INSTANCE.Write(writer, value.TopP)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.Model)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.ResponseFormatJson)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.System)
}

type FfiDestroyerCompletionRequest struct{}

func (_ FfiDestroyerCompletionRequest) Destroy(value CompletionRequest) {
	value.Destroy()
}

// The result of a non-streaming chat completion.
//
// `content` is the empty string when the provider returned no text (e.g.
// the model emitted only tool calls). `finish_reason` is the empty string
// when the provider didn't report one.
type CompletionResponse struct {
	Content      string
	ToolCalls    []ToolCall
	FinishReason string
	Model        string
	Usage        TokenUsage
}

func (r *CompletionResponse) Destroy() {
	FfiDestroyerString{}.Destroy(r.Content)
	FfiDestroyerSequenceToolCall{}.Destroy(r.ToolCalls)
	FfiDestroyerString{}.Destroy(r.FinishReason)
	FfiDestroyerString{}.Destroy(r.Model)
	FfiDestroyerTokenUsage{}.Destroy(r.Usage)
}

type FfiConverterCompletionResponse struct{}

var FfiConverterCompletionResponseINSTANCE = FfiConverterCompletionResponse{}

func (c FfiConverterCompletionResponse) Lift(rb RustBufferI) CompletionResponse {
	return LiftFromRustBuffer[CompletionResponse](c, rb)
}

func (c FfiConverterCompletionResponse) Read(reader io.Reader) CompletionResponse {
	return CompletionResponse{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterSequenceToolCallINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterTokenUsageINSTANCE.Read(reader),
	}
}

func (c FfiConverterCompletionResponse) Lower(value CompletionResponse) C.RustBuffer {
	return LowerIntoRustBuffer[CompletionResponse](c, value)
}

func (c FfiConverterCompletionResponse) LowerExternal(value CompletionResponse) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[CompletionResponse](c, value))
}

func (c FfiConverterCompletionResponse) Write(writer io.Writer, value CompletionResponse) {
	FfiConverterStringINSTANCE.Write(writer, value.Content)
	FfiConverterSequenceToolCallINSTANCE.Write(writer, value.ToolCalls)
	FfiConverterStringINSTANCE.Write(writer, value.FinishReason)
	FfiConverterStringINSTANCE.Write(writer, value.Model)
	FfiConverterTokenUsageINSTANCE.Write(writer, value.Usage)
}

type FfiDestroyerCompletionResponse struct{}

func (_ FfiDestroyerCompletionResponse) Destroy(value CompletionResponse) {
	value.Destroy()
}

// Embedding-role defaults. V1 composes only `base`.
type EmbeddingProviderDefaults struct {
	Base *BaseProviderDefaults
}

func (r *EmbeddingProviderDefaults) Destroy() {
	FfiDestroyerOptionalBaseProviderDefaults{}.Destroy(r.Base)
}

type FfiConverterEmbeddingProviderDefaults struct{}

var FfiConverterEmbeddingProviderDefaultsINSTANCE = FfiConverterEmbeddingProviderDefaults{}

func (c FfiConverterEmbeddingProviderDefaults) Lift(rb RustBufferI) EmbeddingProviderDefaults {
	return LiftFromRustBuffer[EmbeddingProviderDefaults](c, rb)
}

func (c FfiConverterEmbeddingProviderDefaults) Read(reader io.Reader) EmbeddingProviderDefaults {
	return EmbeddingProviderDefaults{
		FfiConverterOptionalBaseProviderDefaultsINSTANCE.Read(reader),
	}
}

func (c FfiConverterEmbeddingProviderDefaults) Lower(value EmbeddingProviderDefaults) C.RustBuffer {
	return LowerIntoRustBuffer[EmbeddingProviderDefaults](c, value)
}

func (c FfiConverterEmbeddingProviderDefaults) LowerExternal(value EmbeddingProviderDefaults) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[EmbeddingProviderDefaults](c, value))
}

func (c FfiConverterEmbeddingProviderDefaults) Write(writer io.Writer, value EmbeddingProviderDefaults) {
	FfiConverterOptionalBaseProviderDefaultsINSTANCE.Write(writer, value.Base)
}

type FfiDestroyerEmbeddingProviderDefaults struct{}

func (_ FfiDestroyerEmbeddingProviderDefaults) Destroy(value EmbeddingProviderDefaults) {
	value.Destroy()
}

// Response from an embedding model.
//
// `embeddings[i]` is the vector for the `i`-th input string. Vectors are
// `f64` for FFI uniformity (UniFFI doesn't expose `f32`); upstream `f32`
// values widen losslessly.
type EmbeddingResponse struct {
	Embeddings [][]float64
	Model      string
	Usage      TokenUsage
}

func (r *EmbeddingResponse) Destroy() {
	FfiDestroyerSequenceSequenceFloat64{}.Destroy(r.Embeddings)
	FfiDestroyerString{}.Destroy(r.Model)
	FfiDestroyerTokenUsage{}.Destroy(r.Usage)
}

type FfiConverterEmbeddingResponse struct{}

var FfiConverterEmbeddingResponseINSTANCE = FfiConverterEmbeddingResponse{}

func (c FfiConverterEmbeddingResponse) Lift(rb RustBufferI) EmbeddingResponse {
	return LiftFromRustBuffer[EmbeddingResponse](c, rb)
}

func (c FfiConverterEmbeddingResponse) Read(reader io.Reader) EmbeddingResponse {
	return EmbeddingResponse{
		FfiConverterSequenceSequenceFloat64INSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterTokenUsageINSTANCE.Read(reader),
	}
}

func (c FfiConverterEmbeddingResponse) Lower(value EmbeddingResponse) C.RustBuffer {
	return LowerIntoRustBuffer[EmbeddingResponse](c, value)
}

func (c FfiConverterEmbeddingResponse) LowerExternal(value EmbeddingResponse) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[EmbeddingResponse](c, value))
}

func (c FfiConverterEmbeddingResponse) Write(writer io.Writer, value EmbeddingResponse) {
	FfiConverterSequenceSequenceFloat64INSTANCE.Write(writer, value.Embeddings)
	FfiConverterStringINSTANCE.Write(writer, value.Model)
	FfiConverterTokenUsageINSTANCE.Write(writer, value.Usage)
}

type FfiDestroyerEmbeddingResponse struct{}

func (_ FfiDestroyerEmbeddingResponse) Destroy(value EmbeddingResponse) {
	value.Destroy()
}

// Event crossed across the FFI boundary.
//
// `event_type` is a free-form string naming the event class (e.g.
// `"StartEvent"`, `"StopEvent"`, `"MyCustomEvent"`). `data_json` is a
// JSON-encoded payload. Foreign-language wrappers typically marshal these
// to/from native types (Go structs, Swift `Codable`, Kotlin `@Serializable`,
// Ruby hashes) just outside this module's boundary.
type Event struct {
	EventType string
	DataJson  string
}

func (r *Event) Destroy() {
	FfiDestroyerString{}.Destroy(r.EventType)
	FfiDestroyerString{}.Destroy(r.DataJson)
}

type FfiConverterEvent struct{}

var FfiConverterEventINSTANCE = FfiConverterEvent{}

func (c FfiConverterEvent) Lift(rb RustBufferI) Event {
	return LiftFromRustBuffer[Event](c, rb)
}

func (c FfiConverterEvent) Read(reader io.Reader) Event {
	return Event{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
	}
}

func (c FfiConverterEvent) Lower(value Event) C.RustBuffer {
	return LowerIntoRustBuffer[Event](c, value)
}

func (c FfiConverterEvent) LowerExternal(value Event) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[Event](c, value))
}

func (c FfiConverterEvent) Write(writer io.Writer, value Event) {
	FfiConverterStringINSTANCE.Write(writer, value.EventType)
	FfiConverterStringINSTANCE.Write(writer, value.DataJson)
}

type FfiDestroyerEvent struct{}

func (_ FfiDestroyerEvent) Destroy(value Event) {
	value.Destroy()
}

// A single generated 3D model with optional mesh metadata.
type Generated3DModel struct {
	Media         MediaOutput
	VertexCount   *uint64
	FaceCount     *uint64
	HasTextures   bool
	HasAnimations bool
}

func (r *Generated3DModel) Destroy() {
	FfiDestroyerMediaOutput{}.Destroy(r.Media)
	FfiDestroyerOptionalUint64{}.Destroy(r.VertexCount)
	FfiDestroyerOptionalUint64{}.Destroy(r.FaceCount)
	FfiDestroyerBool{}.Destroy(r.HasTextures)
	FfiDestroyerBool{}.Destroy(r.HasAnimations)
}

type FfiConverterGenerated3DModel struct{}

var FfiConverterGenerated3DModelINSTANCE = FfiConverterGenerated3DModel{}

func (c FfiConverterGenerated3DModel) Lift(rb RustBufferI) Generated3DModel {
	return LiftFromRustBuffer[Generated3DModel](c, rb)
}

func (c FfiConverterGenerated3DModel) Read(reader io.Reader) Generated3DModel {
	return Generated3DModel{
		FfiConverterMediaOutputINSTANCE.Read(reader),
		FfiConverterOptionalUint64INSTANCE.Read(reader),
		FfiConverterOptionalUint64INSTANCE.Read(reader),
		FfiConverterBoolINSTANCE.Read(reader),
		FfiConverterBoolINSTANCE.Read(reader),
	}
}

func (c FfiConverterGenerated3DModel) Lower(value Generated3DModel) C.RustBuffer {
	return LowerIntoRustBuffer[Generated3DModel](c, value)
}

func (c FfiConverterGenerated3DModel) LowerExternal(value Generated3DModel) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[Generated3DModel](c, value))
}

func (c FfiConverterGenerated3DModel) Write(writer io.Writer, value Generated3DModel) {
	FfiConverterMediaOutputINSTANCE.Write(writer, value.Media)
	FfiConverterOptionalUint64INSTANCE.Write(writer, value.VertexCount)
	FfiConverterOptionalUint64INSTANCE.Write(writer, value.FaceCount)
	FfiConverterBoolINSTANCE.Write(writer, value.HasTextures)
	FfiConverterBoolINSTANCE.Write(writer, value.HasAnimations)
}

type FfiDestroyerGenerated3DModel struct{}

func (_ FfiDestroyerGenerated3DModel) Destroy(value Generated3DModel) {
	value.Destroy()
}

// A single generated audio clip with optional metadata.
type GeneratedAudio struct {
	Media           MediaOutput
	DurationSeconds *float32
	SampleRate      *uint32
	// Number of channels, if known. UniFFI doesn't have a `u8` distinct from
	// `u32`, so the upstream `Option<u8>` widens to `Option<u32>`.
	Channels *uint32
}

func (r *GeneratedAudio) Destroy() {
	FfiDestroyerMediaOutput{}.Destroy(r.Media)
	FfiDestroyerOptionalFloat32{}.Destroy(r.DurationSeconds)
	FfiDestroyerOptionalUint32{}.Destroy(r.SampleRate)
	FfiDestroyerOptionalUint32{}.Destroy(r.Channels)
}

type FfiConverterGeneratedAudio struct{}

var FfiConverterGeneratedAudioINSTANCE = FfiConverterGeneratedAudio{}

func (c FfiConverterGeneratedAudio) Lift(rb RustBufferI) GeneratedAudio {
	return LiftFromRustBuffer[GeneratedAudio](c, rb)
}

func (c FfiConverterGeneratedAudio) Read(reader io.Reader) GeneratedAudio {
	return GeneratedAudio{
		FfiConverterMediaOutputINSTANCE.Read(reader),
		FfiConverterOptionalFloat32INSTANCE.Read(reader),
		FfiConverterOptionalUint32INSTANCE.Read(reader),
		FfiConverterOptionalUint32INSTANCE.Read(reader),
	}
}

func (c FfiConverterGeneratedAudio) Lower(value GeneratedAudio) C.RustBuffer {
	return LowerIntoRustBuffer[GeneratedAudio](c, value)
}

func (c FfiConverterGeneratedAudio) LowerExternal(value GeneratedAudio) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[GeneratedAudio](c, value))
}

func (c FfiConverterGeneratedAudio) Write(writer io.Writer, value GeneratedAudio) {
	FfiConverterMediaOutputINSTANCE.Write(writer, value.Media)
	FfiConverterOptionalFloat32INSTANCE.Write(writer, value.DurationSeconds)
	FfiConverterOptionalUint32INSTANCE.Write(writer, value.SampleRate)
	FfiConverterOptionalUint32INSTANCE.Write(writer, value.Channels)
}

type FfiDestroyerGeneratedAudio struct{}

func (_ FfiDestroyerGeneratedAudio) Destroy(value GeneratedAudio) {
	value.Destroy()
}

// A single generated image with optional dimension metadata.
type GeneratedImage struct {
	Media  MediaOutput
	Width  *uint32
	Height *uint32
}

func (r *GeneratedImage) Destroy() {
	FfiDestroyerMediaOutput{}.Destroy(r.Media)
	FfiDestroyerOptionalUint32{}.Destroy(r.Width)
	FfiDestroyerOptionalUint32{}.Destroy(r.Height)
}

type FfiConverterGeneratedImage struct{}

var FfiConverterGeneratedImageINSTANCE = FfiConverterGeneratedImage{}

func (c FfiConverterGeneratedImage) Lift(rb RustBufferI) GeneratedImage {
	return LiftFromRustBuffer[GeneratedImage](c, rb)
}

func (c FfiConverterGeneratedImage) Read(reader io.Reader) GeneratedImage {
	return GeneratedImage{
		FfiConverterMediaOutputINSTANCE.Read(reader),
		FfiConverterOptionalUint32INSTANCE.Read(reader),
		FfiConverterOptionalUint32INSTANCE.Read(reader),
	}
}

func (c FfiConverterGeneratedImage) Lower(value GeneratedImage) C.RustBuffer {
	return LowerIntoRustBuffer[GeneratedImage](c, value)
}

func (c FfiConverterGeneratedImage) LowerExternal(value GeneratedImage) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[GeneratedImage](c, value))
}

func (c FfiConverterGeneratedImage) Write(writer io.Writer, value GeneratedImage) {
	FfiConverterMediaOutputINSTANCE.Write(writer, value.Media)
	FfiConverterOptionalUint32INSTANCE.Write(writer, value.Width)
	FfiConverterOptionalUint32INSTANCE.Write(writer, value.Height)
}

type FfiDestroyerGeneratedImage struct{}

func (_ FfiDestroyerGeneratedImage) Destroy(value GeneratedImage) {
	value.Destroy()
}

// A single generated video with optional metadata.
type GeneratedVideo struct {
	Media           MediaOutput
	Width           *uint32
	Height          *uint32
	DurationSeconds *float32
	Fps             *float32
}

func (r *GeneratedVideo) Destroy() {
	FfiDestroyerMediaOutput{}.Destroy(r.Media)
	FfiDestroyerOptionalUint32{}.Destroy(r.Width)
	FfiDestroyerOptionalUint32{}.Destroy(r.Height)
	FfiDestroyerOptionalFloat32{}.Destroy(r.DurationSeconds)
	FfiDestroyerOptionalFloat32{}.Destroy(r.Fps)
}

type FfiConverterGeneratedVideo struct{}

var FfiConverterGeneratedVideoINSTANCE = FfiConverterGeneratedVideo{}

func (c FfiConverterGeneratedVideo) Lift(rb RustBufferI) GeneratedVideo {
	return LiftFromRustBuffer[GeneratedVideo](c, rb)
}

func (c FfiConverterGeneratedVideo) Read(reader io.Reader) GeneratedVideo {
	return GeneratedVideo{
		FfiConverterMediaOutputINSTANCE.Read(reader),
		FfiConverterOptionalUint32INSTANCE.Read(reader),
		FfiConverterOptionalUint32INSTANCE.Read(reader),
		FfiConverterOptionalFloat32INSTANCE.Read(reader),
		FfiConverterOptionalFloat32INSTANCE.Read(reader),
	}
}

func (c FfiConverterGeneratedVideo) Lower(value GeneratedVideo) C.RustBuffer {
	return LowerIntoRustBuffer[GeneratedVideo](c, value)
}

func (c FfiConverterGeneratedVideo) LowerExternal(value GeneratedVideo) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[GeneratedVideo](c, value))
}

func (c FfiConverterGeneratedVideo) Write(writer io.Writer, value GeneratedVideo) {
	FfiConverterMediaOutputINSTANCE.Write(writer, value.Media)
	FfiConverterOptionalUint32INSTANCE.Write(writer, value.Width)
	FfiConverterOptionalUint32INSTANCE.Write(writer, value.Height)
	FfiConverterOptionalFloat32INSTANCE.Write(writer, value.DurationSeconds)
	FfiConverterOptionalFloat32INSTANCE.Write(writer, value.Fps)
}

type FfiDestroyerGeneratedVideo struct{}

func (_ FfiDestroyerGeneratedVideo) Destroy(value GeneratedVideo) {
	value.Destroy()
}

// The result of an image-generation call.
//
// `images[i].kind` is always `"image"`. `data_base64` contains either the
// raw base64 bytes (when the upstream `MediaOutput.base64` field is
// populated) or the URL string (when only `MediaOutput.url` is set);
// callers must inspect `mime_type` and treat the field accordingly.
type ImageGenResult struct {
	Images []Media
}

func (r *ImageGenResult) Destroy() {
	FfiDestroyerSequenceMedia{}.Destroy(r.Images)
}

type FfiConverterImageGenResult struct{}

var FfiConverterImageGenResultINSTANCE = FfiConverterImageGenResult{}

func (c FfiConverterImageGenResult) Lift(rb RustBufferI) ImageGenResult {
	return LiftFromRustBuffer[ImageGenResult](c, rb)
}

func (c FfiConverterImageGenResult) Read(reader io.Reader) ImageGenResult {
	return ImageGenResult{
		FfiConverterSequenceMediaINSTANCE.Read(reader),
	}
}

func (c FfiConverterImageGenResult) Lower(value ImageGenResult) C.RustBuffer {
	return LowerIntoRustBuffer[ImageGenResult](c, value)
}

func (c FfiConverterImageGenResult) LowerExternal(value ImageGenResult) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[ImageGenResult](c, value))
}

func (c FfiConverterImageGenResult) Write(writer io.Writer, value ImageGenResult) {
	FfiConverterSequenceMediaINSTANCE.Write(writer, value.Images)
}

type FfiDestroyerImageGenResult struct{}

func (_ FfiDestroyerImageGenResult) Destroy(value ImageGenResult) {
	value.Destroy()
}

type ImageGenerationProviderDefaults struct {
	Base *BaseProviderDefaults
}

func (r *ImageGenerationProviderDefaults) Destroy() {
	FfiDestroyerOptionalBaseProviderDefaults{}.Destroy(r.Base)
}

type FfiConverterImageGenerationProviderDefaults struct{}

var FfiConverterImageGenerationProviderDefaultsINSTANCE = FfiConverterImageGenerationProviderDefaults{}

func (c FfiConverterImageGenerationProviderDefaults) Lift(rb RustBufferI) ImageGenerationProviderDefaults {
	return LiftFromRustBuffer[ImageGenerationProviderDefaults](c, rb)
}

func (c FfiConverterImageGenerationProviderDefaults) Read(reader io.Reader) ImageGenerationProviderDefaults {
	return ImageGenerationProviderDefaults{
		FfiConverterOptionalBaseProviderDefaultsINSTANCE.Read(reader),
	}
}

func (c FfiConverterImageGenerationProviderDefaults) Lower(value ImageGenerationProviderDefaults) C.RustBuffer {
	return LowerIntoRustBuffer[ImageGenerationProviderDefaults](c, value)
}

func (c FfiConverterImageGenerationProviderDefaults) LowerExternal(value ImageGenerationProviderDefaults) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[ImageGenerationProviderDefaults](c, value))
}

func (c FfiConverterImageGenerationProviderDefaults) Write(writer io.Writer, value ImageGenerationProviderDefaults) {
	FfiConverterOptionalBaseProviderDefaultsINSTANCE.Write(writer, value.Base)
}

type FfiDestroyerImageGenerationProviderDefaults struct{}

func (_ FfiDestroyerImageGenerationProviderDefaults) Destroy(value ImageGenerationProviderDefaults) {
	value.Destroy()
}

// Request to generate images from a text prompt.
type ImageRequest struct {
	Prompt         string
	NegativePrompt *string
	Width          *uint32
	Height         *uint32
	NumImages      *uint32
	Model          *string
	// JSON-encoded provider-specific parameters. Empty string is `null`.
	Parameters string
}

func (r *ImageRequest) Destroy() {
	FfiDestroyerString{}.Destroy(r.Prompt)
	FfiDestroyerOptionalString{}.Destroy(r.NegativePrompt)
	FfiDestroyerOptionalUint32{}.Destroy(r.Width)
	FfiDestroyerOptionalUint32{}.Destroy(r.Height)
	FfiDestroyerOptionalUint32{}.Destroy(r.NumImages)
	FfiDestroyerOptionalString{}.Destroy(r.Model)
	FfiDestroyerString{}.Destroy(r.Parameters)
}

type FfiConverterImageRequest struct{}

var FfiConverterImageRequestINSTANCE = FfiConverterImageRequest{}

func (c FfiConverterImageRequest) Lift(rb RustBufferI) ImageRequest {
	return LiftFromRustBuffer[ImageRequest](c, rb)
}

func (c FfiConverterImageRequest) Read(reader io.Reader) ImageRequest {
	return ImageRequest{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterOptionalUint32INSTANCE.Read(reader),
		FfiConverterOptionalUint32INSTANCE.Read(reader),
		FfiConverterOptionalUint32INSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
	}
}

func (c FfiConverterImageRequest) Lower(value ImageRequest) C.RustBuffer {
	return LowerIntoRustBuffer[ImageRequest](c, value)
}

func (c FfiConverterImageRequest) LowerExternal(value ImageRequest) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[ImageRequest](c, value))
}

func (c FfiConverterImageRequest) Write(writer io.Writer, value ImageRequest) {
	FfiConverterStringINSTANCE.Write(writer, value.Prompt)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.NegativePrompt)
	FfiConverterOptionalUint32INSTANCE.Write(writer, value.Width)
	FfiConverterOptionalUint32INSTANCE.Write(writer, value.Height)
	FfiConverterOptionalUint32INSTANCE.Write(writer, value.NumImages)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.Model)
	FfiConverterStringINSTANCE.Write(writer, value.Parameters)
}

type FfiDestroyerImageRequest struct{}

func (_ FfiDestroyerImageRequest) Destroy(value ImageRequest) {
	value.Destroy()
}

// Result of an image generation or upscale operation.
type ImageResult struct {
	Images     []GeneratedImage
	Timing     RequestTiming
	Cost       *float64
	Usage      *TokenUsage
	ImageCount uint32
	Metadata   string
}

func (r *ImageResult) Destroy() {
	FfiDestroyerSequenceGeneratedImage{}.Destroy(r.Images)
	FfiDestroyerRequestTiming{}.Destroy(r.Timing)
	FfiDestroyerOptionalFloat64{}.Destroy(r.Cost)
	FfiDestroyerOptionalTokenUsage{}.Destroy(r.Usage)
	FfiDestroyerUint32{}.Destroy(r.ImageCount)
	FfiDestroyerString{}.Destroy(r.Metadata)
}

type FfiConverterImageResult struct{}

var FfiConverterImageResultINSTANCE = FfiConverterImageResult{}

func (c FfiConverterImageResult) Lift(rb RustBufferI) ImageResult {
	return LiftFromRustBuffer[ImageResult](c, rb)
}

func (c FfiConverterImageResult) Read(reader io.Reader) ImageResult {
	return ImageResult{
		FfiConverterSequenceGeneratedImageINSTANCE.Read(reader),
		FfiConverterRequestTimingINSTANCE.Read(reader),
		FfiConverterOptionalFloat64INSTANCE.Read(reader),
		FfiConverterOptionalTokenUsageINSTANCE.Read(reader),
		FfiConverterUint32INSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
	}
}

func (c FfiConverterImageResult) Lower(value ImageResult) C.RustBuffer {
	return LowerIntoRustBuffer[ImageResult](c, value)
}

func (c FfiConverterImageResult) LowerExternal(value ImageResult) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[ImageResult](c, value))
}

func (c FfiConverterImageResult) Write(writer io.Writer, value ImageResult) {
	FfiConverterSequenceGeneratedImageINSTANCE.Write(writer, value.Images)
	FfiConverterRequestTimingINSTANCE.Write(writer, value.Timing)
	FfiConverterOptionalFloat64INSTANCE.Write(writer, value.Cost)
	FfiConverterOptionalTokenUsageINSTANCE.Write(writer, value.Usage)
	FfiConverterUint32INSTANCE.Write(writer, value.ImageCount)
	FfiConverterStringINSTANCE.Write(writer, value.Metadata)
}

type FfiDestroyerImageResult struct{}

func (_ FfiDestroyerImageResult) Destroy(value ImageResult) {
	value.Destroy()
}

type ImageUpscaleProviderDefaults struct {
	Base *BaseProviderDefaults
}

func (r *ImageUpscaleProviderDefaults) Destroy() {
	FfiDestroyerOptionalBaseProviderDefaults{}.Destroy(r.Base)
}

type FfiConverterImageUpscaleProviderDefaults struct{}

var FfiConverterImageUpscaleProviderDefaultsINSTANCE = FfiConverterImageUpscaleProviderDefaults{}

func (c FfiConverterImageUpscaleProviderDefaults) Lift(rb RustBufferI) ImageUpscaleProviderDefaults {
	return LiftFromRustBuffer[ImageUpscaleProviderDefaults](c, rb)
}

func (c FfiConverterImageUpscaleProviderDefaults) Read(reader io.Reader) ImageUpscaleProviderDefaults {
	return ImageUpscaleProviderDefaults{
		FfiConverterOptionalBaseProviderDefaultsINSTANCE.Read(reader),
	}
}

func (c FfiConverterImageUpscaleProviderDefaults) Lower(value ImageUpscaleProviderDefaults) C.RustBuffer {
	return LowerIntoRustBuffer[ImageUpscaleProviderDefaults](c, value)
}

func (c FfiConverterImageUpscaleProviderDefaults) LowerExternal(value ImageUpscaleProviderDefaults) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[ImageUpscaleProviderDefaults](c, value))
}

func (c FfiConverterImageUpscaleProviderDefaults) Write(writer io.Writer, value ImageUpscaleProviderDefaults) {
	FfiConverterOptionalBaseProviderDefaultsINSTANCE.Write(writer, value.Base)
}

type FfiDestroyerImageUpscaleProviderDefaults struct{}

func (_ FfiDestroyerImageUpscaleProviderDefaults) Destroy(value ImageUpscaleProviderDefaults) {
	value.Destroy()
}

// Simple key/value pair for extra HTTP headers and query parameters.
//
// Upstream uses `Vec<(String, String)>`; UniFFI Records can't represent
// raw tuples, so we lift them into a named record.
type KeyValue struct {
	Key   string
	Value string
}

func (r *KeyValue) Destroy() {
	FfiDestroyerString{}.Destroy(r.Key)
	FfiDestroyerString{}.Destroy(r.Value)
}

type FfiConverterKeyValue struct{}

var FfiConverterKeyValueINSTANCE = FfiConverterKeyValue{}

func (c FfiConverterKeyValue) Lift(rb RustBufferI) KeyValue {
	return LiftFromRustBuffer[KeyValue](c, rb)
}

func (c FfiConverterKeyValue) Read(reader io.Reader) KeyValue {
	return KeyValue{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
	}
}

func (c FfiConverterKeyValue) Lower(value KeyValue) C.RustBuffer {
	return LowerIntoRustBuffer[KeyValue](c, value)
}

func (c FfiConverterKeyValue) LowerExternal(value KeyValue) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[KeyValue](c, value))
}

func (c FfiConverterKeyValue) Write(writer io.Writer, value KeyValue) {
	FfiConverterStringINSTANCE.Write(writer, value.Key)
	FfiConverterStringINSTANCE.Write(writer, value.Value)
}

type FfiDestroyerKeyValue struct{}

func (_ FfiDestroyerKeyValue) Destroy(value KeyValue) {
	value.Destroy()
}

// Multimodal media attached to a [`ChatMessage`].
//
// `kind` selects the part type and is one of `"image"`, `"audio"`, `"video"`.
// `mime_type` is the IANA MIME (`"image/png"`, `"audio/mp3"`, ...).
// `data_base64` carries the raw bytes base64-encoded; for URL-backed media,
// set `data_base64` to the empty string and put the URL in `mime_type` is
// **not** correct — instead, callers wanting URL inputs should base64-encode
// the fetched bytes. (URL passthrough is intentionally elided here to keep
// the FFI surface single-shape; providers that need URL fidelity can be
// reached via `providers.rs`.)
type Media struct {
	Kind       string
	MimeType   string
	DataBase64 string
}

func (r *Media) Destroy() {
	FfiDestroyerString{}.Destroy(r.Kind)
	FfiDestroyerString{}.Destroy(r.MimeType)
	FfiDestroyerString{}.Destroy(r.DataBase64)
}

type FfiConverterMedia struct{}

var FfiConverterMediaINSTANCE = FfiConverterMedia{}

func (c FfiConverterMedia) Lift(rb RustBufferI) Media {
	return LiftFromRustBuffer[Media](c, rb)
}

func (c FfiConverterMedia) Read(reader io.Reader) Media {
	return Media{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
	}
}

func (c FfiConverterMedia) Lower(value Media) C.RustBuffer {
	return LowerIntoRustBuffer[Media](c, value)
}

func (c FfiConverterMedia) LowerExternal(value Media) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[Media](c, value))
}

func (c FfiConverterMedia) Write(writer io.Writer, value Media) {
	FfiConverterStringINSTANCE.Write(writer, value.Kind)
	FfiConverterStringINSTANCE.Write(writer, value.MimeType)
	FfiConverterStringINSTANCE.Write(writer, value.DataBase64)
}

type FfiDestroyerMedia struct{}

func (_ FfiDestroyerMedia) Destroy(value Media) {
	value.Destroy()
}

// A single piece of generated media (image, video, audio, 3D, ...).
//
// At least one of `url`, `base64`, or `raw_content` is populated.
// `media_type` is the canonical MIME string (`"image/png"`, `"video/mp4"`,
// `"model/gltf-binary"`, ...). Unknown MIMEs map back to
// [`CoreMediaType::Other`].
type MediaOutput struct {
	Url        *string
	Base64     *string
	RawContent *string
	// MIME type string (e.g. `"image/png"`).
	MediaType string
	FileSize  *uint64
	// JSON-encoded arbitrary metadata. Empty string round-trips to `null`.
	Metadata string
}

func (r *MediaOutput) Destroy() {
	FfiDestroyerOptionalString{}.Destroy(r.Url)
	FfiDestroyerOptionalString{}.Destroy(r.Base64)
	FfiDestroyerOptionalString{}.Destroy(r.RawContent)
	FfiDestroyerString{}.Destroy(r.MediaType)
	FfiDestroyerOptionalUint64{}.Destroy(r.FileSize)
	FfiDestroyerString{}.Destroy(r.Metadata)
}

type FfiConverterMediaOutput struct{}

var FfiConverterMediaOutputINSTANCE = FfiConverterMediaOutput{}

func (c FfiConverterMediaOutput) Lift(rb RustBufferI) MediaOutput {
	return LiftFromRustBuffer[MediaOutput](c, rb)
}

func (c FfiConverterMediaOutput) Read(reader io.Reader) MediaOutput {
	return MediaOutput{
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterOptionalUint64INSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
	}
}

func (c FfiConverterMediaOutput) Lower(value MediaOutput) C.RustBuffer {
	return LowerIntoRustBuffer[MediaOutput](c, value)
}

func (c FfiConverterMediaOutput) LowerExternal(value MediaOutput) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[MediaOutput](c, value))
}

func (c FfiConverterMediaOutput) Write(writer io.Writer, value MediaOutput) {
	FfiConverterOptionalStringINSTANCE.Write(writer, value.Url)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.Base64)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.RawContent)
	FfiConverterStringINSTANCE.Write(writer, value.MediaType)
	FfiConverterOptionalUint64INSTANCE.Write(writer, value.FileSize)
	FfiConverterStringINSTANCE.Write(writer, value.Metadata)
}

type FfiDestroyerMediaOutput struct{}

func (_ FfiDestroyerMediaOutput) Destroy(value MediaOutput) {
	value.Destroy()
}

// Request to generate music or sound effects.
type MusicRequest struct {
	Prompt          string
	DurationSeconds *float32
	Model           *string
	Parameters      string
}

func (r *MusicRequest) Destroy() {
	FfiDestroyerString{}.Destroy(r.Prompt)
	FfiDestroyerOptionalFloat32{}.Destroy(r.DurationSeconds)
	FfiDestroyerOptionalString{}.Destroy(r.Model)
	FfiDestroyerString{}.Destroy(r.Parameters)
}

type FfiConverterMusicRequest struct{}

var FfiConverterMusicRequestINSTANCE = FfiConverterMusicRequest{}

func (c FfiConverterMusicRequest) Lift(rb RustBufferI) MusicRequest {
	return LiftFromRustBuffer[MusicRequest](c, rb)
}

func (c FfiConverterMusicRequest) Read(reader io.Reader) MusicRequest {
	return MusicRequest{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterOptionalFloat32INSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
	}
}

func (c FfiConverterMusicRequest) Lower(value MusicRequest) C.RustBuffer {
	return LowerIntoRustBuffer[MusicRequest](c, value)
}

func (c FfiConverterMusicRequest) LowerExternal(value MusicRequest) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[MusicRequest](c, value))
}

func (c FfiConverterMusicRequest) Write(writer io.Writer, value MusicRequest) {
	FfiConverterStringINSTANCE.Write(writer, value.Prompt)
	FfiConverterOptionalFloat32INSTANCE.Write(writer, value.DurationSeconds)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.Model)
	FfiConverterStringINSTANCE.Write(writer, value.Parameters)
}

type FfiDestroyerMusicRequest struct{}

func (_ FfiDestroyerMusicRequest) Destroy(value MusicRequest) {
	value.Destroy()
}

// Configuration for an OpenAI-compatible provider backend.
//
// Used by the [`ApiProtocol::OpenAi`] variant.
type OpenAiCompatConfig struct {
	// Human-readable name for this provider (used in logs and model info).
	ProviderName string
	// Base URL for the API (e.g. `https://api.openai.com/v1`).
	BaseUrl string
	// API key. May be empty if the provider doesn't require auth.
	ApiKey string
	// Default model to use if a request doesn't override it.
	DefaultModel string
	// How to send the API key.
	AuthMethod AuthMethod
	// Extra HTTP headers to include in every request.
	ExtraHeaders []KeyValue
	// Query parameters to include in every request (e.g. Azure's
	// `api-version`).
	QueryParams []KeyValue
	// Whether this provider supports the `/models` listing endpoint.
	SupportsModelListing bool
}

func (r *OpenAiCompatConfig) Destroy() {
	FfiDestroyerString{}.Destroy(r.ProviderName)
	FfiDestroyerString{}.Destroy(r.BaseUrl)
	FfiDestroyerString{}.Destroy(r.ApiKey)
	FfiDestroyerString{}.Destroy(r.DefaultModel)
	FfiDestroyerAuthMethod{}.Destroy(r.AuthMethod)
	FfiDestroyerSequenceKeyValue{}.Destroy(r.ExtraHeaders)
	FfiDestroyerSequenceKeyValue{}.Destroy(r.QueryParams)
	FfiDestroyerBool{}.Destroy(r.SupportsModelListing)
}

type FfiConverterOpenAiCompatConfig struct{}

var FfiConverterOpenAiCompatConfigINSTANCE = FfiConverterOpenAiCompatConfig{}

func (c FfiConverterOpenAiCompatConfig) Lift(rb RustBufferI) OpenAiCompatConfig {
	return LiftFromRustBuffer[OpenAiCompatConfig](c, rb)
}

func (c FfiConverterOpenAiCompatConfig) Read(reader io.Reader) OpenAiCompatConfig {
	return OpenAiCompatConfig{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterAuthMethodINSTANCE.Read(reader),
		FfiConverterSequenceKeyValueINSTANCE.Read(reader),
		FfiConverterSequenceKeyValueINSTANCE.Read(reader),
		FfiConverterBoolINSTANCE.Read(reader),
	}
}

func (c FfiConverterOpenAiCompatConfig) Lower(value OpenAiCompatConfig) C.RustBuffer {
	return LowerIntoRustBuffer[OpenAiCompatConfig](c, value)
}

func (c FfiConverterOpenAiCompatConfig) LowerExternal(value OpenAiCompatConfig) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[OpenAiCompatConfig](c, value))
}

func (c FfiConverterOpenAiCompatConfig) Write(writer io.Writer, value OpenAiCompatConfig) {
	FfiConverterStringINSTANCE.Write(writer, value.ProviderName)
	FfiConverterStringINSTANCE.Write(writer, value.BaseUrl)
	FfiConverterStringINSTANCE.Write(writer, value.ApiKey)
	FfiConverterStringINSTANCE.Write(writer, value.DefaultModel)
	FfiConverterAuthMethodINSTANCE.Write(writer, value.AuthMethod)
	FfiConverterSequenceKeyValueINSTANCE.Write(writer, value.ExtraHeaders)
	FfiConverterSequenceKeyValueINSTANCE.Write(writer, value.QueryParams)
	FfiConverterBoolINSTANCE.Write(writer, value.SupportsModelListing)
}

type FfiDestroyerOpenAiCompatConfig struct{}

func (_ FfiDestroyerOpenAiCompatConfig) Destroy(value OpenAiCompatConfig) {
	value.Destroy()
}

// A serialized representation of a queued event captured in a checkpoint.
//
// Mirrors [`blazen_persist::SerializedEvent`]. The `data_json` field is
// the JSON-encoded payload of the original event (the upstream type
// stores a `serde_json::Value`, which is not a UniFFI-supported wire
// type — JSON strings cross cleanly instead).
type PersistedEvent struct {
	// The event type identifier (e.g. `"blazen::StartEvent"`).
	EventType string
	// The event payload, JSON-encoded. Decode with the host language's
	// standard JSON library on the foreign side.
	DataJson string
}

func (r *PersistedEvent) Destroy() {
	FfiDestroyerString{}.Destroy(r.EventType)
	FfiDestroyerString{}.Destroy(r.DataJson)
}

type FfiConverterPersistedEvent struct{}

var FfiConverterPersistedEventINSTANCE = FfiConverterPersistedEvent{}

func (c FfiConverterPersistedEvent) Lift(rb RustBufferI) PersistedEvent {
	return LiftFromRustBuffer[PersistedEvent](c, rb)
}

func (c FfiConverterPersistedEvent) Read(reader io.Reader) PersistedEvent {
	return PersistedEvent{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
	}
}

func (c FfiConverterPersistedEvent) Lower(value PersistedEvent) C.RustBuffer {
	return LowerIntoRustBuffer[PersistedEvent](c, value)
}

func (c FfiConverterPersistedEvent) LowerExternal(value PersistedEvent) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[PersistedEvent](c, value))
}

func (c FfiConverterPersistedEvent) Write(writer io.Writer, value PersistedEvent) {
	FfiConverterStringINSTANCE.Write(writer, value.EventType)
	FfiConverterStringINSTANCE.Write(writer, value.DataJson)
}

type FfiDestroyerPersistedEvent struct{}

func (_ FfiDestroyerPersistedEvent) Destroy(value PersistedEvent) {
	value.Destroy()
}

// Timing metadata for a compute request.
//
// All three counters are optional; a `None` value means "the provider did
// not report this timing breakdown" rather than zero.
type RequestTiming struct {
	QueueMs     *uint64
	ExecutionMs *uint64
	TotalMs     *uint64
}

func (r *RequestTiming) Destroy() {
	FfiDestroyerOptionalUint64{}.Destroy(r.QueueMs)
	FfiDestroyerOptionalUint64{}.Destroy(r.ExecutionMs)
	FfiDestroyerOptionalUint64{}.Destroy(r.TotalMs)
}

type FfiConverterRequestTiming struct{}

var FfiConverterRequestTimingINSTANCE = FfiConverterRequestTiming{}

func (c FfiConverterRequestTiming) Lift(rb RustBufferI) RequestTiming {
	return LiftFromRustBuffer[RequestTiming](c, rb)
}

func (c FfiConverterRequestTiming) Read(reader io.Reader) RequestTiming {
	return RequestTiming{
		FfiConverterOptionalUint64INSTANCE.Read(reader),
		FfiConverterOptionalUint64INSTANCE.Read(reader),
		FfiConverterOptionalUint64INSTANCE.Read(reader),
	}
}

func (c FfiConverterRequestTiming) Lower(value RequestTiming) C.RustBuffer {
	return LowerIntoRustBuffer[RequestTiming](c, value)
}

func (c FfiConverterRequestTiming) LowerExternal(value RequestTiming) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[RequestTiming](c, value))
}

func (c FfiConverterRequestTiming) Write(writer io.Writer, value RequestTiming) {
	FfiConverterOptionalUint64INSTANCE.Write(writer, value.QueueMs)
	FfiConverterOptionalUint64INSTANCE.Write(writer, value.ExecutionMs)
	FfiConverterOptionalUint64INSTANCE.Write(writer, value.TotalMs)
}

type FfiDestroyerRequestTiming struct{}

func (_ FfiDestroyerRequestTiming) Destroy(value RequestTiming) {
	value.Destroy()
}

// Request to generate speech from text (TTS).
type SpeechRequest struct {
	Text       string
	Voice      *string
	VoiceUrl   *string
	Language   *string
	Speed      *float32
	Model      *string
	Parameters string
}

func (r *SpeechRequest) Destroy() {
	FfiDestroyerString{}.Destroy(r.Text)
	FfiDestroyerOptionalString{}.Destroy(r.Voice)
	FfiDestroyerOptionalString{}.Destroy(r.VoiceUrl)
	FfiDestroyerOptionalString{}.Destroy(r.Language)
	FfiDestroyerOptionalFloat32{}.Destroy(r.Speed)
	FfiDestroyerOptionalString{}.Destroy(r.Model)
	FfiDestroyerString{}.Destroy(r.Parameters)
}

type FfiConverterSpeechRequest struct{}

var FfiConverterSpeechRequestINSTANCE = FfiConverterSpeechRequest{}

func (c FfiConverterSpeechRequest) Lift(rb RustBufferI) SpeechRequest {
	return LiftFromRustBuffer[SpeechRequest](c, rb)
}

func (c FfiConverterSpeechRequest) Read(reader io.Reader) SpeechRequest {
	return SpeechRequest{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterOptionalFloat32INSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
	}
}

func (c FfiConverterSpeechRequest) Lower(value SpeechRequest) C.RustBuffer {
	return LowerIntoRustBuffer[SpeechRequest](c, value)
}

func (c FfiConverterSpeechRequest) LowerExternal(value SpeechRequest) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[SpeechRequest](c, value))
}

func (c FfiConverterSpeechRequest) Write(writer io.Writer, value SpeechRequest) {
	FfiConverterStringINSTANCE.Write(writer, value.Text)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.Voice)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.VoiceUrl)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.Language)
	FfiConverterOptionalFloat32INSTANCE.Write(writer, value.Speed)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.Model)
	FfiConverterStringINSTANCE.Write(writer, value.Parameters)
}

type FfiDestroyerSpeechRequest struct{}

func (_ FfiDestroyerSpeechRequest) Destroy(value SpeechRequest) {
	value.Destroy()
}

// A single chunk from a streaming chat completion.
//
// Chunks arrive in order. `content_delta` is the incremental text since the
// last chunk (empty when the chunk carries only tool-call deltas or
// reasoning trace). `tool_calls` is the latest known set of tool
// invocations — upstream providers may emit tool-call deltas across
// multiple chunks, so consumers should treat each chunk's `tool_calls` as a
// snapshot rather than an append-only list.
//
// `is_final` is set on the last content-bearing chunk before
// [`CompletionStreamSink::on_done`] fires. It is a UI hint (e.g. "stop
// showing the typing cursor") and does not replace `on_done` for cleanup.
type StreamChunk struct {
	// Incremental text delta since the previous chunk. Empty when the
	// chunk carries only tool-call deltas, reasoning trace, citations, or
	// artifacts.
	ContentDelta string
	// Tool-call snapshot for this chunk. May grow as the provider streams
	// tool-call arguments piecewise; consumers should replace, not append.
	ToolCalls []ToolCall
	// True if this is the final content-bearing chunk before `on_done`.
	// Hint only — `on_done` is the authoritative completion signal.
	IsFinal bool
}

func (r *StreamChunk) Destroy() {
	FfiDestroyerString{}.Destroy(r.ContentDelta)
	FfiDestroyerSequenceToolCall{}.Destroy(r.ToolCalls)
	FfiDestroyerBool{}.Destroy(r.IsFinal)
}

type FfiConverterStreamChunk struct{}

var FfiConverterStreamChunkINSTANCE = FfiConverterStreamChunk{}

func (c FfiConverterStreamChunk) Lift(rb RustBufferI) StreamChunk {
	return LiftFromRustBuffer[StreamChunk](c, rb)
}

func (c FfiConverterStreamChunk) Read(reader io.Reader) StreamChunk {
	return StreamChunk{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterSequenceToolCallINSTANCE.Read(reader),
		FfiConverterBoolINSTANCE.Read(reader),
	}
}

func (c FfiConverterStreamChunk) Lower(value StreamChunk) C.RustBuffer {
	return LowerIntoRustBuffer[StreamChunk](c, value)
}

func (c FfiConverterStreamChunk) LowerExternal(value StreamChunk) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[StreamChunk](c, value))
}

func (c FfiConverterStreamChunk) Write(writer io.Writer, value StreamChunk) {
	FfiConverterStringINSTANCE.Write(writer, value.ContentDelta)
	FfiConverterSequenceToolCallINSTANCE.Write(writer, value.ToolCalls)
	FfiConverterBoolINSTANCE.Write(writer, value.IsFinal)
}

type FfiDestroyerStreamChunk struct{}

func (_ FfiDestroyerStreamChunk) Destroy(value StreamChunk) {
	value.Destroy()
}

// The result of a speech-to-text transcription call.
//
// `language` is the empty string when the provider didn't report a
// detected language. `duration_ms` reflects the upstream
// [`RequestTiming::total_ms`](blazen_llm::RequestTiming) — zero when the
// backend didn't measure it.
type SttResult struct {
	Transcript string
	Language   string
	DurationMs uint64
}

func (r *SttResult) Destroy() {
	FfiDestroyerString{}.Destroy(r.Transcript)
	FfiDestroyerString{}.Destroy(r.Language)
	FfiDestroyerUint64{}.Destroy(r.DurationMs)
}

type FfiConverterSttResult struct{}

var FfiConverterSttResultINSTANCE = FfiConverterSttResult{}

func (c FfiConverterSttResult) Lift(rb RustBufferI) SttResult {
	return LiftFromRustBuffer[SttResult](c, rb)
}

func (c FfiConverterSttResult) Read(reader io.Reader) SttResult {
	return SttResult{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterUint64INSTANCE.Read(reader),
	}
}

func (c FfiConverterSttResult) Lower(value SttResult) C.RustBuffer {
	return LowerIntoRustBuffer[SttResult](c, value)
}

func (c FfiConverterSttResult) LowerExternal(value SttResult) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[SttResult](c, value))
}

func (c FfiConverterSttResult) Write(writer io.Writer, value SttResult) {
	FfiConverterStringINSTANCE.Write(writer, value.Transcript)
	FfiConverterStringINSTANCE.Write(writer, value.Language)
	FfiConverterUint64INSTANCE.Write(writer, value.DurationMs)
}

type FfiDestroyerSttResult struct{}

func (_ FfiDestroyerSttResult) Destroy(value SttResult) {
	value.Destroy()
}

type ThreeDProviderDefaults struct {
	Base *BaseProviderDefaults
}

func (r *ThreeDProviderDefaults) Destroy() {
	FfiDestroyerOptionalBaseProviderDefaults{}.Destroy(r.Base)
}

type FfiConverterThreeDProviderDefaults struct{}

var FfiConverterThreeDProviderDefaultsINSTANCE = FfiConverterThreeDProviderDefaults{}

func (c FfiConverterThreeDProviderDefaults) Lift(rb RustBufferI) ThreeDProviderDefaults {
	return LiftFromRustBuffer[ThreeDProviderDefaults](c, rb)
}

func (c FfiConverterThreeDProviderDefaults) Read(reader io.Reader) ThreeDProviderDefaults {
	return ThreeDProviderDefaults{
		FfiConverterOptionalBaseProviderDefaultsINSTANCE.Read(reader),
	}
}

func (c FfiConverterThreeDProviderDefaults) Lower(value ThreeDProviderDefaults) C.RustBuffer {
	return LowerIntoRustBuffer[ThreeDProviderDefaults](c, value)
}

func (c FfiConverterThreeDProviderDefaults) LowerExternal(value ThreeDProviderDefaults) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[ThreeDProviderDefaults](c, value))
}

func (c FfiConverterThreeDProviderDefaults) Write(writer io.Writer, value ThreeDProviderDefaults) {
	FfiConverterOptionalBaseProviderDefaultsINSTANCE.Write(writer, value.Base)
}

type FfiDestroyerThreeDProviderDefaults struct{}

func (_ FfiDestroyerThreeDProviderDefaults) Destroy(value ThreeDProviderDefaults) {
	value.Destroy()
}

// Request to generate a 3D model.
type ThreeDRequest struct {
	Prompt     string
	ImageUrl   *string
	Format     *string
	Model      *string
	Parameters string
}

func (r *ThreeDRequest) Destroy() {
	FfiDestroyerString{}.Destroy(r.Prompt)
	FfiDestroyerOptionalString{}.Destroy(r.ImageUrl)
	FfiDestroyerOptionalString{}.Destroy(r.Format)
	FfiDestroyerOptionalString{}.Destroy(r.Model)
	FfiDestroyerString{}.Destroy(r.Parameters)
}

type FfiConverterThreeDRequest struct{}

var FfiConverterThreeDRequestINSTANCE = FfiConverterThreeDRequest{}

func (c FfiConverterThreeDRequest) Lift(rb RustBufferI) ThreeDRequest {
	return LiftFromRustBuffer[ThreeDRequest](c, rb)
}

func (c FfiConverterThreeDRequest) Read(reader io.Reader) ThreeDRequest {
	return ThreeDRequest{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
	}
}

func (c FfiConverterThreeDRequest) Lower(value ThreeDRequest) C.RustBuffer {
	return LowerIntoRustBuffer[ThreeDRequest](c, value)
}

func (c FfiConverterThreeDRequest) LowerExternal(value ThreeDRequest) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[ThreeDRequest](c, value))
}

func (c FfiConverterThreeDRequest) Write(writer io.Writer, value ThreeDRequest) {
	FfiConverterStringINSTANCE.Write(writer, value.Prompt)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.ImageUrl)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.Format)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.Model)
	FfiConverterStringINSTANCE.Write(writer, value.Parameters)
}

type FfiDestroyerThreeDRequest struct{}

func (_ FfiDestroyerThreeDRequest) Destroy(value ThreeDRequest) {
	value.Destroy()
}

// Result of a 3D model generation operation.
type ThreeDResult struct {
	Models   []Generated3DModel
	Timing   RequestTiming
	Cost     *float64
	Usage    *TokenUsage
	Metadata string
}

func (r *ThreeDResult) Destroy() {
	FfiDestroyerSequenceGenerated3DModel{}.Destroy(r.Models)
	FfiDestroyerRequestTiming{}.Destroy(r.Timing)
	FfiDestroyerOptionalFloat64{}.Destroy(r.Cost)
	FfiDestroyerOptionalTokenUsage{}.Destroy(r.Usage)
	FfiDestroyerString{}.Destroy(r.Metadata)
}

type FfiConverterThreeDResult struct{}

var FfiConverterThreeDResultINSTANCE = FfiConverterThreeDResult{}

func (c FfiConverterThreeDResult) Lift(rb RustBufferI) ThreeDResult {
	return LiftFromRustBuffer[ThreeDResult](c, rb)
}

func (c FfiConverterThreeDResult) Read(reader io.Reader) ThreeDResult {
	return ThreeDResult{
		FfiConverterSequenceGenerated3DModelINSTANCE.Read(reader),
		FfiConverterRequestTimingINSTANCE.Read(reader),
		FfiConverterOptionalFloat64INSTANCE.Read(reader),
		FfiConverterOptionalTokenUsageINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
	}
}

func (c FfiConverterThreeDResult) Lower(value ThreeDResult) C.RustBuffer {
	return LowerIntoRustBuffer[ThreeDResult](c, value)
}

func (c FfiConverterThreeDResult) LowerExternal(value ThreeDResult) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[ThreeDResult](c, value))
}

func (c FfiConverterThreeDResult) Write(writer io.Writer, value ThreeDResult) {
	FfiConverterSequenceGenerated3DModelINSTANCE.Write(writer, value.Models)
	FfiConverterRequestTimingINSTANCE.Write(writer, value.Timing)
	FfiConverterOptionalFloat64INSTANCE.Write(writer, value.Cost)
	FfiConverterOptionalTokenUsageINSTANCE.Write(writer, value.Usage)
	FfiConverterStringINSTANCE.Write(writer, value.Metadata)
}

type FfiDestroyerThreeDResult struct{}

func (_ FfiDestroyerThreeDResult) Destroy(value ThreeDResult) {
	value.Destroy()
}

// Token usage statistics for a completion or embedding request.
//
// Every counter is `u64` for FFI uniformity. Upstream `u32` values widen
// losslessly. Zero means either "the provider didn't report this counter"
// or "the counter is genuinely zero" — the wire format does not distinguish.
type TokenUsage struct {
	PromptTokens      uint64
	CompletionTokens  uint64
	TotalTokens       uint64
	CachedInputTokens uint64
	ReasoningTokens   uint64
}

func (r *TokenUsage) Destroy() {
	FfiDestroyerUint64{}.Destroy(r.PromptTokens)
	FfiDestroyerUint64{}.Destroy(r.CompletionTokens)
	FfiDestroyerUint64{}.Destroy(r.TotalTokens)
	FfiDestroyerUint64{}.Destroy(r.CachedInputTokens)
	FfiDestroyerUint64{}.Destroy(r.ReasoningTokens)
}

type FfiConverterTokenUsage struct{}

var FfiConverterTokenUsageINSTANCE = FfiConverterTokenUsage{}

func (c FfiConverterTokenUsage) Lift(rb RustBufferI) TokenUsage {
	return LiftFromRustBuffer[TokenUsage](c, rb)
}

func (c FfiConverterTokenUsage) Read(reader io.Reader) TokenUsage {
	return TokenUsage{
		FfiConverterUint64INSTANCE.Read(reader),
		FfiConverterUint64INSTANCE.Read(reader),
		FfiConverterUint64INSTANCE.Read(reader),
		FfiConverterUint64INSTANCE.Read(reader),
		FfiConverterUint64INSTANCE.Read(reader),
	}
}

func (c FfiConverterTokenUsage) Lower(value TokenUsage) C.RustBuffer {
	return LowerIntoRustBuffer[TokenUsage](c, value)
}

func (c FfiConverterTokenUsage) LowerExternal(value TokenUsage) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[TokenUsage](c, value))
}

func (c FfiConverterTokenUsage) Write(writer io.Writer, value TokenUsage) {
	FfiConverterUint64INSTANCE.Write(writer, value.PromptTokens)
	FfiConverterUint64INSTANCE.Write(writer, value.CompletionTokens)
	FfiConverterUint64INSTANCE.Write(writer, value.TotalTokens)
	FfiConverterUint64INSTANCE.Write(writer, value.CachedInputTokens)
	FfiConverterUint64INSTANCE.Write(writer, value.ReasoningTokens)
}

type FfiDestroyerTokenUsage struct{}

func (_ FfiDestroyerTokenUsage) Destroy(value TokenUsage) {
	value.Destroy()
}

// A tool that the model may invoke during a completion.
type Tool struct {
	Name        string
	Description string
	// JSON Schema describing the tool's input parameters. Stored as a string
	// on the wire; foreign callers serialize their native schema dict/struct
	// to JSON just before constructing the [`Tool`].
	ParametersJson string
}

func (r *Tool) Destroy() {
	FfiDestroyerString{}.Destroy(r.Name)
	FfiDestroyerString{}.Destroy(r.Description)
	FfiDestroyerString{}.Destroy(r.ParametersJson)
}

type FfiConverterTool struct{}

var FfiConverterToolINSTANCE = FfiConverterTool{}

func (c FfiConverterTool) Lift(rb RustBufferI) Tool {
	return LiftFromRustBuffer[Tool](c, rb)
}

func (c FfiConverterTool) Read(reader io.Reader) Tool {
	return Tool{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
	}
}

func (c FfiConverterTool) Lower(value Tool) C.RustBuffer {
	return LowerIntoRustBuffer[Tool](c, value)
}

func (c FfiConverterTool) LowerExternal(value Tool) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[Tool](c, value))
}

func (c FfiConverterTool) Write(writer io.Writer, value Tool) {
	FfiConverterStringINSTANCE.Write(writer, value.Name)
	FfiConverterStringINSTANCE.Write(writer, value.Description)
	FfiConverterStringINSTANCE.Write(writer, value.ParametersJson)
}

type FfiDestroyerTool struct{}

func (_ FfiDestroyerTool) Destroy(value Tool) {
	value.Destroy()
}

// A tool invocation requested by the model.
type ToolCall struct {
	Id   string
	Name string
	// JSON-encoded arguments object. Foreign callers parse this with their
	// native JSON library to access the tool's input parameters.
	ArgumentsJson string
}

func (r *ToolCall) Destroy() {
	FfiDestroyerString{}.Destroy(r.Id)
	FfiDestroyerString{}.Destroy(r.Name)
	FfiDestroyerString{}.Destroy(r.ArgumentsJson)
}

type FfiConverterToolCall struct{}

var FfiConverterToolCallINSTANCE = FfiConverterToolCall{}

func (c FfiConverterToolCall) Lift(rb RustBufferI) ToolCall {
	return LiftFromRustBuffer[ToolCall](c, rb)
}

func (c FfiConverterToolCall) Read(reader io.Reader) ToolCall {
	return ToolCall{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
	}
}

func (c FfiConverterToolCall) Lower(value ToolCall) C.RustBuffer {
	return LowerIntoRustBuffer[ToolCall](c, value)
}

func (c FfiConverterToolCall) LowerExternal(value ToolCall) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[ToolCall](c, value))
}

func (c FfiConverterToolCall) Write(writer io.Writer, value ToolCall) {
	FfiConverterStringINSTANCE.Write(writer, value.Id)
	FfiConverterStringINSTANCE.Write(writer, value.Name)
	FfiConverterStringINSTANCE.Write(writer, value.ArgumentsJson)
}

type FfiDestroyerToolCall struct{}

func (_ FfiDestroyerToolCall) Destroy(value ToolCall) {
	value.Destroy()
}

type TranscriptionProviderDefaults struct {
	Base *BaseProviderDefaults
}

func (r *TranscriptionProviderDefaults) Destroy() {
	FfiDestroyerOptionalBaseProviderDefaults{}.Destroy(r.Base)
}

type FfiConverterTranscriptionProviderDefaults struct{}

var FfiConverterTranscriptionProviderDefaultsINSTANCE = FfiConverterTranscriptionProviderDefaults{}

func (c FfiConverterTranscriptionProviderDefaults) Lift(rb RustBufferI) TranscriptionProviderDefaults {
	return LiftFromRustBuffer[TranscriptionProviderDefaults](c, rb)
}

func (c FfiConverterTranscriptionProviderDefaults) Read(reader io.Reader) TranscriptionProviderDefaults {
	return TranscriptionProviderDefaults{
		FfiConverterOptionalBaseProviderDefaultsINSTANCE.Read(reader),
	}
}

func (c FfiConverterTranscriptionProviderDefaults) Lower(value TranscriptionProviderDefaults) C.RustBuffer {
	return LowerIntoRustBuffer[TranscriptionProviderDefaults](c, value)
}

func (c FfiConverterTranscriptionProviderDefaults) LowerExternal(value TranscriptionProviderDefaults) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[TranscriptionProviderDefaults](c, value))
}

func (c FfiConverterTranscriptionProviderDefaults) Write(writer io.Writer, value TranscriptionProviderDefaults) {
	FfiConverterOptionalBaseProviderDefaultsINSTANCE.Write(writer, value.Base)
}

type FfiDestroyerTranscriptionProviderDefaults struct{}

func (_ FfiDestroyerTranscriptionProviderDefaults) Destroy(value TranscriptionProviderDefaults) {
	value.Destroy()
}

// Request to transcribe audio to text.
//
// `audio_source_json` carries the full upstream
// [`MediaSource`](blazen_llm::types::MediaSource) as a JSON-encoded string
// (one of `{"type":"url","url":"..."}`, `{"type":"base64","data":"..."}`,
// `{"type":"file","path":"..."}`, `{"type":"provider_file",...}`,
// `{"type":"handle",...}`). `None` falls back to using `audio_url`.
type TranscriptionRequest struct {
	AudioUrl string
	// JSON-encoded `MediaSource`. When `Some`, takes precedence over
	// `audio_url`. See module docs for the wire shape.
	AudioSourceJson *string
	Language        *string
	Diarize         bool
	Model           *string
	Parameters      string
}

func (r *TranscriptionRequest) Destroy() {
	FfiDestroyerString{}.Destroy(r.AudioUrl)
	FfiDestroyerOptionalString{}.Destroy(r.AudioSourceJson)
	FfiDestroyerOptionalString{}.Destroy(r.Language)
	FfiDestroyerBool{}.Destroy(r.Diarize)
	FfiDestroyerOptionalString{}.Destroy(r.Model)
	FfiDestroyerString{}.Destroy(r.Parameters)
}

type FfiConverterTranscriptionRequest struct{}

var FfiConverterTranscriptionRequestINSTANCE = FfiConverterTranscriptionRequest{}

func (c FfiConverterTranscriptionRequest) Lift(rb RustBufferI) TranscriptionRequest {
	return LiftFromRustBuffer[TranscriptionRequest](c, rb)
}

func (c FfiConverterTranscriptionRequest) Read(reader io.Reader) TranscriptionRequest {
	return TranscriptionRequest{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterBoolINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
	}
}

func (c FfiConverterTranscriptionRequest) Lower(value TranscriptionRequest) C.RustBuffer {
	return LowerIntoRustBuffer[TranscriptionRequest](c, value)
}

func (c FfiConverterTranscriptionRequest) LowerExternal(value TranscriptionRequest) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[TranscriptionRequest](c, value))
}

func (c FfiConverterTranscriptionRequest) Write(writer io.Writer, value TranscriptionRequest) {
	FfiConverterStringINSTANCE.Write(writer, value.AudioUrl)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.AudioSourceJson)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.Language)
	FfiConverterBoolINSTANCE.Write(writer, value.Diarize)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.Model)
	FfiConverterStringINSTANCE.Write(writer, value.Parameters)
}

type FfiDestroyerTranscriptionRequest struct{}

func (_ FfiDestroyerTranscriptionRequest) Destroy(value TranscriptionRequest) {
	value.Destroy()
}

// Result of a transcription operation.
type TranscriptionResult struct {
	Text         string
	Segments     []TranscriptionSegment
	Language     *string
	Timing       RequestTiming
	Cost         *float64
	Usage        *TokenUsage
	AudioSeconds float64
	Metadata     string
}

func (r *TranscriptionResult) Destroy() {
	FfiDestroyerString{}.Destroy(r.Text)
	FfiDestroyerSequenceTranscriptionSegment{}.Destroy(r.Segments)
	FfiDestroyerOptionalString{}.Destroy(r.Language)
	FfiDestroyerRequestTiming{}.Destroy(r.Timing)
	FfiDestroyerOptionalFloat64{}.Destroy(r.Cost)
	FfiDestroyerOptionalTokenUsage{}.Destroy(r.Usage)
	FfiDestroyerFloat64{}.Destroy(r.AudioSeconds)
	FfiDestroyerString{}.Destroy(r.Metadata)
}

type FfiConverterTranscriptionResult struct{}

var FfiConverterTranscriptionResultINSTANCE = FfiConverterTranscriptionResult{}

func (c FfiConverterTranscriptionResult) Lift(rb RustBufferI) TranscriptionResult {
	return LiftFromRustBuffer[TranscriptionResult](c, rb)
}

func (c FfiConverterTranscriptionResult) Read(reader io.Reader) TranscriptionResult {
	return TranscriptionResult{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterSequenceTranscriptionSegmentINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterRequestTimingINSTANCE.Read(reader),
		FfiConverterOptionalFloat64INSTANCE.Read(reader),
		FfiConverterOptionalTokenUsageINSTANCE.Read(reader),
		FfiConverterFloat64INSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
	}
}

func (c FfiConverterTranscriptionResult) Lower(value TranscriptionResult) C.RustBuffer {
	return LowerIntoRustBuffer[TranscriptionResult](c, value)
}

func (c FfiConverterTranscriptionResult) LowerExternal(value TranscriptionResult) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[TranscriptionResult](c, value))
}

func (c FfiConverterTranscriptionResult) Write(writer io.Writer, value TranscriptionResult) {
	FfiConverterStringINSTANCE.Write(writer, value.Text)
	FfiConverterSequenceTranscriptionSegmentINSTANCE.Write(writer, value.Segments)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.Language)
	FfiConverterRequestTimingINSTANCE.Write(writer, value.Timing)
	FfiConverterOptionalFloat64INSTANCE.Write(writer, value.Cost)
	FfiConverterOptionalTokenUsageINSTANCE.Write(writer, value.Usage)
	FfiConverterFloat64INSTANCE.Write(writer, value.AudioSeconds)
	FfiConverterStringINSTANCE.Write(writer, value.Metadata)
}

type FfiDestroyerTranscriptionResult struct{}

func (_ FfiDestroyerTranscriptionResult) Destroy(value TranscriptionResult) {
	value.Destroy()
}

// A single segment within a transcription.
type TranscriptionSegment struct {
	Text    string
	Start   float64
	End     float64
	Speaker *string
}

func (r *TranscriptionSegment) Destroy() {
	FfiDestroyerString{}.Destroy(r.Text)
	FfiDestroyerFloat64{}.Destroy(r.Start)
	FfiDestroyerFloat64{}.Destroy(r.End)
	FfiDestroyerOptionalString{}.Destroy(r.Speaker)
}

type FfiConverterTranscriptionSegment struct{}

var FfiConverterTranscriptionSegmentINSTANCE = FfiConverterTranscriptionSegment{}

func (c FfiConverterTranscriptionSegment) Lift(rb RustBufferI) TranscriptionSegment {
	return LiftFromRustBuffer[TranscriptionSegment](c, rb)
}

func (c FfiConverterTranscriptionSegment) Read(reader io.Reader) TranscriptionSegment {
	return TranscriptionSegment{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterFloat64INSTANCE.Read(reader),
		FfiConverterFloat64INSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
	}
}

func (c FfiConverterTranscriptionSegment) Lower(value TranscriptionSegment) C.RustBuffer {
	return LowerIntoRustBuffer[TranscriptionSegment](c, value)
}

func (c FfiConverterTranscriptionSegment) LowerExternal(value TranscriptionSegment) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[TranscriptionSegment](c, value))
}

func (c FfiConverterTranscriptionSegment) Write(writer io.Writer, value TranscriptionSegment) {
	FfiConverterStringINSTANCE.Write(writer, value.Text)
	FfiConverterFloat64INSTANCE.Write(writer, value.Start)
	FfiConverterFloat64INSTANCE.Write(writer, value.End)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.Speaker)
}

type FfiDestroyerTranscriptionSegment struct{}

func (_ FfiDestroyerTranscriptionSegment) Destroy(value TranscriptionSegment) {
	value.Destroy()
}

// The result of a text-to-speech synthesis call.
//
// `audio_base64` is the empty string when the upstream provider returned a
// URL only (the URL travels in the `data_base64` slot of a downstream
// [`Media`] when callers route through [`crate::llm::Media`]; pure TTS
// callers should detect the empty `audio_base64` and fall back to fetching
// the URL themselves). `mime_type` reflects the upstream
// [`MediaType`](blazen_llm::MediaType); `duration_ms` is zero when the
// provider didn't report timing.
type TtsResult struct {
	AudioBase64 string
	MimeType    string
	DurationMs  uint64
}

func (r *TtsResult) Destroy() {
	FfiDestroyerString{}.Destroy(r.AudioBase64)
	FfiDestroyerString{}.Destroy(r.MimeType)
	FfiDestroyerUint64{}.Destroy(r.DurationMs)
}

type FfiConverterTtsResult struct{}

var FfiConverterTtsResultINSTANCE = FfiConverterTtsResult{}

func (c FfiConverterTtsResult) Lift(rb RustBufferI) TtsResult {
	return LiftFromRustBuffer[TtsResult](c, rb)
}

func (c FfiConverterTtsResult) Read(reader io.Reader) TtsResult {
	return TtsResult{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterUint64INSTANCE.Read(reader),
	}
}

func (c FfiConverterTtsResult) Lower(value TtsResult) C.RustBuffer {
	return LowerIntoRustBuffer[TtsResult](c, value)
}

func (c FfiConverterTtsResult) LowerExternal(value TtsResult) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[TtsResult](c, value))
}

func (c FfiConverterTtsResult) Write(writer io.Writer, value TtsResult) {
	FfiConverterStringINSTANCE.Write(writer, value.AudioBase64)
	FfiConverterStringINSTANCE.Write(writer, value.MimeType)
	FfiConverterUint64INSTANCE.Write(writer, value.DurationMs)
}

type FfiDestroyerTtsResult struct{}

func (_ FfiDestroyerTtsResult) Destroy(value TtsResult) {
	value.Destroy()
}

// Request to upscale an image.
type UpscaleRequest struct {
	ImageUrl   string
	Scale      float32
	Model      *string
	Parameters string
}

func (r *UpscaleRequest) Destroy() {
	FfiDestroyerString{}.Destroy(r.ImageUrl)
	FfiDestroyerFloat32{}.Destroy(r.Scale)
	FfiDestroyerOptionalString{}.Destroy(r.Model)
	FfiDestroyerString{}.Destroy(r.Parameters)
}

type FfiConverterUpscaleRequest struct{}

var FfiConverterUpscaleRequestINSTANCE = FfiConverterUpscaleRequest{}

func (c FfiConverterUpscaleRequest) Lift(rb RustBufferI) UpscaleRequest {
	return LiftFromRustBuffer[UpscaleRequest](c, rb)
}

func (c FfiConverterUpscaleRequest) Read(reader io.Reader) UpscaleRequest {
	return UpscaleRequest{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterFloat32INSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
	}
}

func (c FfiConverterUpscaleRequest) Lower(value UpscaleRequest) C.RustBuffer {
	return LowerIntoRustBuffer[UpscaleRequest](c, value)
}

func (c FfiConverterUpscaleRequest) LowerExternal(value UpscaleRequest) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[UpscaleRequest](c, value))
}

func (c FfiConverterUpscaleRequest) Write(writer io.Writer, value UpscaleRequest) {
	FfiConverterStringINSTANCE.Write(writer, value.ImageUrl)
	FfiConverterFloat32INSTANCE.Write(writer, value.Scale)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.Model)
	FfiConverterStringINSTANCE.Write(writer, value.Parameters)
}

type FfiDestroyerUpscaleRequest struct{}

func (_ FfiDestroyerUpscaleRequest) Destroy(value UpscaleRequest) {
	value.Destroy()
}

type VideoProviderDefaults struct {
	Base *BaseProviderDefaults
}

func (r *VideoProviderDefaults) Destroy() {
	FfiDestroyerOptionalBaseProviderDefaults{}.Destroy(r.Base)
}

type FfiConverterVideoProviderDefaults struct{}

var FfiConverterVideoProviderDefaultsINSTANCE = FfiConverterVideoProviderDefaults{}

func (c FfiConverterVideoProviderDefaults) Lift(rb RustBufferI) VideoProviderDefaults {
	return LiftFromRustBuffer[VideoProviderDefaults](c, rb)
}

func (c FfiConverterVideoProviderDefaults) Read(reader io.Reader) VideoProviderDefaults {
	return VideoProviderDefaults{
		FfiConverterOptionalBaseProviderDefaultsINSTANCE.Read(reader),
	}
}

func (c FfiConverterVideoProviderDefaults) Lower(value VideoProviderDefaults) C.RustBuffer {
	return LowerIntoRustBuffer[VideoProviderDefaults](c, value)
}

func (c FfiConverterVideoProviderDefaults) LowerExternal(value VideoProviderDefaults) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[VideoProviderDefaults](c, value))
}

func (c FfiConverterVideoProviderDefaults) Write(writer io.Writer, value VideoProviderDefaults) {
	FfiConverterOptionalBaseProviderDefaultsINSTANCE.Write(writer, value.Base)
}

type FfiDestroyerVideoProviderDefaults struct{}

func (_ FfiDestroyerVideoProviderDefaults) Destroy(value VideoProviderDefaults) {
	value.Destroy()
}

// Request to generate a video.
type VideoRequest struct {
	Prompt          string
	ImageUrl        *string
	DurationSeconds *float32
	NegativePrompt  *string
	Width           *uint32
	Height          *uint32
	Model           *string
	Parameters      string
}

func (r *VideoRequest) Destroy() {
	FfiDestroyerString{}.Destroy(r.Prompt)
	FfiDestroyerOptionalString{}.Destroy(r.ImageUrl)
	FfiDestroyerOptionalFloat32{}.Destroy(r.DurationSeconds)
	FfiDestroyerOptionalString{}.Destroy(r.NegativePrompt)
	FfiDestroyerOptionalUint32{}.Destroy(r.Width)
	FfiDestroyerOptionalUint32{}.Destroy(r.Height)
	FfiDestroyerOptionalString{}.Destroy(r.Model)
	FfiDestroyerString{}.Destroy(r.Parameters)
}

type FfiConverterVideoRequest struct{}

var FfiConverterVideoRequestINSTANCE = FfiConverterVideoRequest{}

func (c FfiConverterVideoRequest) Lift(rb RustBufferI) VideoRequest {
	return LiftFromRustBuffer[VideoRequest](c, rb)
}

func (c FfiConverterVideoRequest) Read(reader io.Reader) VideoRequest {
	return VideoRequest{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterOptionalFloat32INSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterOptionalUint32INSTANCE.Read(reader),
		FfiConverterOptionalUint32INSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
	}
}

func (c FfiConverterVideoRequest) Lower(value VideoRequest) C.RustBuffer {
	return LowerIntoRustBuffer[VideoRequest](c, value)
}

func (c FfiConverterVideoRequest) LowerExternal(value VideoRequest) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[VideoRequest](c, value))
}

func (c FfiConverterVideoRequest) Write(writer io.Writer, value VideoRequest) {
	FfiConverterStringINSTANCE.Write(writer, value.Prompt)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.ImageUrl)
	FfiConverterOptionalFloat32INSTANCE.Write(writer, value.DurationSeconds)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.NegativePrompt)
	FfiConverterOptionalUint32INSTANCE.Write(writer, value.Width)
	FfiConverterOptionalUint32INSTANCE.Write(writer, value.Height)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.Model)
	FfiConverterStringINSTANCE.Write(writer, value.Parameters)
}

type FfiDestroyerVideoRequest struct{}

func (_ FfiDestroyerVideoRequest) Destroy(value VideoRequest) {
	value.Destroy()
}

// Result of a video generation operation.
type VideoResult struct {
	Videos       []GeneratedVideo
	Timing       RequestTiming
	Cost         *float64
	Usage        *TokenUsage
	VideoSeconds float64
	Metadata     string
}

func (r *VideoResult) Destroy() {
	FfiDestroyerSequenceGeneratedVideo{}.Destroy(r.Videos)
	FfiDestroyerRequestTiming{}.Destroy(r.Timing)
	FfiDestroyerOptionalFloat64{}.Destroy(r.Cost)
	FfiDestroyerOptionalTokenUsage{}.Destroy(r.Usage)
	FfiDestroyerFloat64{}.Destroy(r.VideoSeconds)
	FfiDestroyerString{}.Destroy(r.Metadata)
}

type FfiConverterVideoResult struct{}

var FfiConverterVideoResultINSTANCE = FfiConverterVideoResult{}

func (c FfiConverterVideoResult) Lift(rb RustBufferI) VideoResult {
	return LiftFromRustBuffer[VideoResult](c, rb)
}

func (c FfiConverterVideoResult) Read(reader io.Reader) VideoResult {
	return VideoResult{
		FfiConverterSequenceGeneratedVideoINSTANCE.Read(reader),
		FfiConverterRequestTimingINSTANCE.Read(reader),
		FfiConverterOptionalFloat64INSTANCE.Read(reader),
		FfiConverterOptionalTokenUsageINSTANCE.Read(reader),
		FfiConverterFloat64INSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
	}
}

func (c FfiConverterVideoResult) Lower(value VideoResult) C.RustBuffer {
	return LowerIntoRustBuffer[VideoResult](c, value)
}

func (c FfiConverterVideoResult) LowerExternal(value VideoResult) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[VideoResult](c, value))
}

func (c FfiConverterVideoResult) Write(writer io.Writer, value VideoResult) {
	FfiConverterSequenceGeneratedVideoINSTANCE.Write(writer, value.Videos)
	FfiConverterRequestTimingINSTANCE.Write(writer, value.Timing)
	FfiConverterOptionalFloat64INSTANCE.Write(writer, value.Cost)
	FfiConverterOptionalTokenUsageINSTANCE.Write(writer, value.Usage)
	FfiConverterFloat64INSTANCE.Write(writer, value.VideoSeconds)
	FfiConverterStringINSTANCE.Write(writer, value.Metadata)
}

type FfiDestroyerVideoResult struct{}

func (_ FfiDestroyerVideoResult) Destroy(value VideoResult) {
	value.Destroy()
}

// Request to clone a voice from one or more reference audio clips.
type VoiceCloneRequest struct {
	Name          string
	ReferenceUrls []string
	Language      *string
	Description   *string
	Parameters    string
}

func (r *VoiceCloneRequest) Destroy() {
	FfiDestroyerString{}.Destroy(r.Name)
	FfiDestroyerSequenceString{}.Destroy(r.ReferenceUrls)
	FfiDestroyerOptionalString{}.Destroy(r.Language)
	FfiDestroyerOptionalString{}.Destroy(r.Description)
	FfiDestroyerString{}.Destroy(r.Parameters)
}

type FfiConverterVoiceCloneRequest struct{}

var FfiConverterVoiceCloneRequestINSTANCE = FfiConverterVoiceCloneRequest{}

func (c FfiConverterVoiceCloneRequest) Lift(rb RustBufferI) VoiceCloneRequest {
	return LiftFromRustBuffer[VoiceCloneRequest](c, rb)
}

func (c FfiConverterVoiceCloneRequest) Read(reader io.Reader) VoiceCloneRequest {
	return VoiceCloneRequest{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterSequenceStringINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
	}
}

func (c FfiConverterVoiceCloneRequest) Lower(value VoiceCloneRequest) C.RustBuffer {
	return LowerIntoRustBuffer[VoiceCloneRequest](c, value)
}

func (c FfiConverterVoiceCloneRequest) LowerExternal(value VoiceCloneRequest) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[VoiceCloneRequest](c, value))
}

func (c FfiConverterVoiceCloneRequest) Write(writer io.Writer, value VoiceCloneRequest) {
	FfiConverterStringINSTANCE.Write(writer, value.Name)
	FfiConverterSequenceStringINSTANCE.Write(writer, value.ReferenceUrls)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.Language)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.Description)
	FfiConverterStringINSTANCE.Write(writer, value.Parameters)
}

type FfiDestroyerVoiceCloneRequest struct{}

func (_ FfiDestroyerVoiceCloneRequest) Destroy(value VoiceCloneRequest) {
	value.Destroy()
}

type VoiceCloningProviderDefaults struct {
	Base *BaseProviderDefaults
}

func (r *VoiceCloningProviderDefaults) Destroy() {
	FfiDestroyerOptionalBaseProviderDefaults{}.Destroy(r.Base)
}

type FfiConverterVoiceCloningProviderDefaults struct{}

var FfiConverterVoiceCloningProviderDefaultsINSTANCE = FfiConverterVoiceCloningProviderDefaults{}

func (c FfiConverterVoiceCloningProviderDefaults) Lift(rb RustBufferI) VoiceCloningProviderDefaults {
	return LiftFromRustBuffer[VoiceCloningProviderDefaults](c, rb)
}

func (c FfiConverterVoiceCloningProviderDefaults) Read(reader io.Reader) VoiceCloningProviderDefaults {
	return VoiceCloningProviderDefaults{
		FfiConverterOptionalBaseProviderDefaultsINSTANCE.Read(reader),
	}
}

func (c FfiConverterVoiceCloningProviderDefaults) Lower(value VoiceCloningProviderDefaults) C.RustBuffer {
	return LowerIntoRustBuffer[VoiceCloningProviderDefaults](c, value)
}

func (c FfiConverterVoiceCloningProviderDefaults) LowerExternal(value VoiceCloningProviderDefaults) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[VoiceCloningProviderDefaults](c, value))
}

func (c FfiConverterVoiceCloningProviderDefaults) Write(writer io.Writer, value VoiceCloningProviderDefaults) {
	FfiConverterOptionalBaseProviderDefaultsINSTANCE.Write(writer, value.Base)
}

type FfiDestroyerVoiceCloningProviderDefaults struct{}

func (_ FfiDestroyerVoiceCloningProviderDefaults) Destroy(value VoiceCloningProviderDefaults) {
	value.Destroy()
}

// A persisted voice identifier returned by a `VoiceCloning` provider.
type VoiceHandle struct {
	Id          string
	Name        string
	Provider    string
	Language    *string
	Description *string
	Metadata    string
}

func (r *VoiceHandle) Destroy() {
	FfiDestroyerString{}.Destroy(r.Id)
	FfiDestroyerString{}.Destroy(r.Name)
	FfiDestroyerString{}.Destroy(r.Provider)
	FfiDestroyerOptionalString{}.Destroy(r.Language)
	FfiDestroyerOptionalString{}.Destroy(r.Description)
	FfiDestroyerString{}.Destroy(r.Metadata)
}

type FfiConverterVoiceHandle struct{}

var FfiConverterVoiceHandleINSTANCE = FfiConverterVoiceHandle{}

func (c FfiConverterVoiceHandle) Lift(rb RustBufferI) VoiceHandle {
	return LiftFromRustBuffer[VoiceHandle](c, rb)
}

func (c FfiConverterVoiceHandle) Read(reader io.Reader) VoiceHandle {
	return VoiceHandle{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
	}
}

func (c FfiConverterVoiceHandle) Lower(value VoiceHandle) C.RustBuffer {
	return LowerIntoRustBuffer[VoiceHandle](c, value)
}

func (c FfiConverterVoiceHandle) LowerExternal(value VoiceHandle) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[VoiceHandle](c, value))
}

func (c FfiConverterVoiceHandle) Write(writer io.Writer, value VoiceHandle) {
	FfiConverterStringINSTANCE.Write(writer, value.Id)
	FfiConverterStringINSTANCE.Write(writer, value.Name)
	FfiConverterStringINSTANCE.Write(writer, value.Provider)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.Language)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.Description)
	FfiConverterStringINSTANCE.Write(writer, value.Metadata)
}

type FfiDestroyerVoiceHandle struct{}

func (_ FfiDestroyerVoiceHandle) Destroy(value VoiceHandle) {
	value.Destroy()
}

// A snapshot of a workflow's state at a point in time.
//
// Foreign-language wrappers typically marshal these to/from native types
// (Go structs, Swift `Codable`, Kotlin `@Serializable`, Ruby hashes) just
// outside this module's boundary.
//
// See the module docs for the upstream field-name mapping.
type WorkflowCheckpoint struct {
	// The name of the workflow that produced this checkpoint.
	WorkflowName string
	// Unique identifier for this workflow run, formatted as a UUID
	// string (`"550e8400-e29b-41d4-a716-446655440000"`).
	RunId string
	// When the checkpoint was created, as Unix-epoch milliseconds.
	TimestampMs uint64
	// Serialized context state, as a JSON object encoded into a string
	// (`"{\"counter\":42}"`). Decode with the host language's JSON library.
	StateJson string
	// Events in the queue at checkpoint time.
	PendingEvents []PersistedEvent
	// Arbitrary metadata attached to this checkpoint, as a JSON object
	// encoded into a string. Decode with the host language's JSON library.
	MetadataJson string
}

func (r *WorkflowCheckpoint) Destroy() {
	FfiDestroyerString{}.Destroy(r.WorkflowName)
	FfiDestroyerString{}.Destroy(r.RunId)
	FfiDestroyerUint64{}.Destroy(r.TimestampMs)
	FfiDestroyerString{}.Destroy(r.StateJson)
	FfiDestroyerSequencePersistedEvent{}.Destroy(r.PendingEvents)
	FfiDestroyerString{}.Destroy(r.MetadataJson)
}

type FfiConverterWorkflowCheckpoint struct{}

var FfiConverterWorkflowCheckpointINSTANCE = FfiConverterWorkflowCheckpoint{}

func (c FfiConverterWorkflowCheckpoint) Lift(rb RustBufferI) WorkflowCheckpoint {
	return LiftFromRustBuffer[WorkflowCheckpoint](c, rb)
}

func (c FfiConverterWorkflowCheckpoint) Read(reader io.Reader) WorkflowCheckpoint {
	return WorkflowCheckpoint{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterUint64INSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterSequencePersistedEventINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
	}
}

func (c FfiConverterWorkflowCheckpoint) Lower(value WorkflowCheckpoint) C.RustBuffer {
	return LowerIntoRustBuffer[WorkflowCheckpoint](c, value)
}

func (c FfiConverterWorkflowCheckpoint) LowerExternal(value WorkflowCheckpoint) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[WorkflowCheckpoint](c, value))
}

func (c FfiConverterWorkflowCheckpoint) Write(writer io.Writer, value WorkflowCheckpoint) {
	FfiConverterStringINSTANCE.Write(writer, value.WorkflowName)
	FfiConverterStringINSTANCE.Write(writer, value.RunId)
	FfiConverterUint64INSTANCE.Write(writer, value.TimestampMs)
	FfiConverterStringINSTANCE.Write(writer, value.StateJson)
	FfiConverterSequencePersistedEventINSTANCE.Write(writer, value.PendingEvents)
	FfiConverterStringINSTANCE.Write(writer, value.MetadataJson)
}

type FfiDestroyerWorkflowCheckpoint struct{}

func (_ FfiDestroyerWorkflowCheckpoint) Destroy(value WorkflowCheckpoint) {
	value.Destroy()
}

// One flattened slot of a workflow execution history, suitable for FFI.
//
// Mirrors the wire shape of an upstream
// [`blazen_telemetry::HistoryEvent`] but collapses the typed
// [`blazen_telemetry::HistoryEventKind`] enum into three flat fields
// (`event_type`, `event_data_json`, plus surface-pulled `step_name`,
// `duration_ms`, `error`) so that Go / Swift / Kotlin / Ruby callers don't
// need to model an open-ended sum type. The full typed payload is always
// available in `event_data_json` as the serde JSON representation.
//
// `workflow_id` is propagated from the enclosing
// [`blazen_telemetry::WorkflowHistory::run_id`] so each entry is
// self-identifying.
type WorkflowHistoryEntry struct {
	// UUID of the workflow run this event belongs to.
	WorkflowId string
	// Step name when the event is step- or LLM-call-scoped; empty otherwise.
	StepName string
	// Variant tag of the upstream `HistoryEventKind` (e.g.
	// `"WorkflowStarted"`, `"StepCompleted"`, `"LlmCallFailed"`).
	EventType string
	// Full serde JSON payload of the upstream `HistoryEventKind` variant —
	// always includes the variant tag plus every typed field.
	EventDataJson string
	// Event timestamp as Unix epoch milliseconds.
	TimestampMs uint64
	// Step / LLM-call duration in milliseconds, when the variant carries it.
	DurationMs *uint64
	// Error message, when the variant is a failure variant.
	Error *string
}

func (r *WorkflowHistoryEntry) Destroy() {
	FfiDestroyerString{}.Destroy(r.WorkflowId)
	FfiDestroyerString{}.Destroy(r.StepName)
	FfiDestroyerString{}.Destroy(r.EventType)
	FfiDestroyerString{}.Destroy(r.EventDataJson)
	FfiDestroyerUint64{}.Destroy(r.TimestampMs)
	FfiDestroyerOptionalUint64{}.Destroy(r.DurationMs)
	FfiDestroyerOptionalString{}.Destroy(r.Error)
}

type FfiConverterWorkflowHistoryEntry struct{}

var FfiConverterWorkflowHistoryEntryINSTANCE = FfiConverterWorkflowHistoryEntry{}

func (c FfiConverterWorkflowHistoryEntry) Lift(rb RustBufferI) WorkflowHistoryEntry {
	return LiftFromRustBuffer[WorkflowHistoryEntry](c, rb)
}

func (c FfiConverterWorkflowHistoryEntry) Read(reader io.Reader) WorkflowHistoryEntry {
	return WorkflowHistoryEntry{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterUint64INSTANCE.Read(reader),
		FfiConverterOptionalUint64INSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
	}
}

func (c FfiConverterWorkflowHistoryEntry) Lower(value WorkflowHistoryEntry) C.RustBuffer {
	return LowerIntoRustBuffer[WorkflowHistoryEntry](c, value)
}

func (c FfiConverterWorkflowHistoryEntry) LowerExternal(value WorkflowHistoryEntry) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[WorkflowHistoryEntry](c, value))
}

func (c FfiConverterWorkflowHistoryEntry) Write(writer io.Writer, value WorkflowHistoryEntry) {
	FfiConverterStringINSTANCE.Write(writer, value.WorkflowId)
	FfiConverterStringINSTANCE.Write(writer, value.StepName)
	FfiConverterStringINSTANCE.Write(writer, value.EventType)
	FfiConverterStringINSTANCE.Write(writer, value.EventDataJson)
	FfiConverterUint64INSTANCE.Write(writer, value.TimestampMs)
	FfiConverterOptionalUint64INSTANCE.Write(writer, value.DurationMs)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.Error)
}

type FfiDestroyerWorkflowHistoryEntry struct{}

func (_ FfiDestroyerWorkflowHistoryEntry) Destroy(value WorkflowHistoryEntry) {
	value.Destroy()
}

// Final result of a workflow run.
type WorkflowResult struct {
	// The terminal event (typically `"StopEvent"`).
	Event Event
	// Total LLM token usage across the run, if any LLM steps ran.
	TotalInputTokens  uint64
	TotalOutputTokens uint64
	// Total cost in USD across the run, if pricing data was available.
	TotalCostUsd float64
}

func (r *WorkflowResult) Destroy() {
	FfiDestroyerEvent{}.Destroy(r.Event)
	FfiDestroyerUint64{}.Destroy(r.TotalInputTokens)
	FfiDestroyerUint64{}.Destroy(r.TotalOutputTokens)
	FfiDestroyerFloat64{}.Destroy(r.TotalCostUsd)
}

type FfiConverterWorkflowResult struct{}

var FfiConverterWorkflowResultINSTANCE = FfiConverterWorkflowResult{}

func (c FfiConverterWorkflowResult) Lift(rb RustBufferI) WorkflowResult {
	return LiftFromRustBuffer[WorkflowResult](c, rb)
}

func (c FfiConverterWorkflowResult) Read(reader io.Reader) WorkflowResult {
	return WorkflowResult{
		FfiConverterEventINSTANCE.Read(reader),
		FfiConverterUint64INSTANCE.Read(reader),
		FfiConverterUint64INSTANCE.Read(reader),
		FfiConverterFloat64INSTANCE.Read(reader),
	}
}

func (c FfiConverterWorkflowResult) Lower(value WorkflowResult) C.RustBuffer {
	return LowerIntoRustBuffer[WorkflowResult](c, value)
}

func (c FfiConverterWorkflowResult) LowerExternal(value WorkflowResult) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[WorkflowResult](c, value))
}

func (c FfiConverterWorkflowResult) Write(writer io.Writer, value WorkflowResult) {
	FfiConverterEventINSTANCE.Write(writer, value.Event)
	FfiConverterUint64INSTANCE.Write(writer, value.TotalInputTokens)
	FfiConverterUint64INSTANCE.Write(writer, value.TotalOutputTokens)
	FfiConverterFloat64INSTANCE.Write(writer, value.TotalCostUsd)
}

type FfiDestroyerWorkflowResult struct{}

func (_ FfiDestroyerWorkflowResult) Destroy(value WorkflowResult) {
	value.Destroy()
}

// Selects how a [`CustomProvider`] talks to its backend for completion
// calls.
//
// - [`ApiProtocol::OpenAi`]: framework handles HTTP, SSE parsing, tool
// calls, retries. The wrapped `OpenAiCompatConfig` supplies the base
// URL, model, optional API key, headers, and query parameters.
// - [`ApiProtocol::Custom`]: framework dispatches every completion
// method to a foreign-implemented [`crate::provider_custom::CustomProvider`]
// trait object. No additional configuration here — the foreign impl owns
// the wire format.
//
// Media-generation calls always go through the foreign-implemented
// `CustomProvider` regardless of which protocol is selected here.
type ApiProtocol interface {
	Destroy()
}

// OpenAI Chat Completions wire format.
type ApiProtocolOpenAi struct {
	Config OpenAiCompatConfig
}

func (e ApiProtocolOpenAi) Destroy() {
	FfiDestroyerOpenAiCompatConfig{}.Destroy(e.Config)
}

// User-defined protocol — handled by a foreign-implemented
// [`crate::provider_custom::CustomProvider`] trait object.
type ApiProtocolCustom struct {
}

func (e ApiProtocolCustom) Destroy() {
}

type FfiConverterApiProtocol struct{}

var FfiConverterApiProtocolINSTANCE = FfiConverterApiProtocol{}

func (c FfiConverterApiProtocol) Lift(rb RustBufferI) ApiProtocol {
	return LiftFromRustBuffer[ApiProtocol](c, rb)
}

func (c FfiConverterApiProtocol) Lower(value ApiProtocol) C.RustBuffer {
	return LowerIntoRustBuffer[ApiProtocol](c, value)
}

func (c FfiConverterApiProtocol) LowerExternal(value ApiProtocol) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[ApiProtocol](c, value))
}
func (FfiConverterApiProtocol) Read(reader io.Reader) ApiProtocol {
	id := readInt32(reader)
	switch id {
	case 1:
		return ApiProtocolOpenAi{
			FfiConverterOpenAiCompatConfigINSTANCE.Read(reader),
		}
	case 2:
		return ApiProtocolCustom{}
	default:
		panic(fmt.Sprintf("invalid enum value %v in FfiConverterApiProtocol.Read()", id))
	}
}

func (FfiConverterApiProtocol) Write(writer io.Writer, value ApiProtocol) {
	switch variant_value := value.(type) {
	case ApiProtocolOpenAi:
		writeInt32(writer, 1)
		FfiConverterOpenAiCompatConfigINSTANCE.Write(writer, variant_value.Config)
	case ApiProtocolCustom:
		writeInt32(writer, 2)
	default:
		_ = variant_value
		panic(fmt.Sprintf("invalid enum value `%v` in FfiConverterApiProtocol.Write", value))
	}
}

type FfiDestroyerApiProtocol struct{}

func (_ FfiDestroyerApiProtocol) Destroy(value ApiProtocol) {
	value.Destroy()
}

// How a [`CustomProvider`] authenticates with an OpenAI-compatible backend.
type AuthMethod interface {
	Destroy()
}

// `Authorization: Bearer <key>` (OpenAI, OpenRouter, Groq, etc.).
type AuthMethodBearer struct {
}

func (e AuthMethodBearer) Destroy() {
}

// A custom header name for the API key (e.g. `x-api-key`).
type AuthMethodApiKeyHeader struct {
	HeaderName string
}

func (e AuthMethodApiKeyHeader) Destroy() {
	FfiDestroyerString{}.Destroy(e.HeaderName)
}

// `api-key: <key>` (Azure OpenAI).
type AuthMethodAzureApiKey struct {
}

func (e AuthMethodAzureApiKey) Destroy() {
}

// `Authorization: Key <key>` (fal.ai).
type AuthMethodKeyPrefix struct {
}

func (e AuthMethodKeyPrefix) Destroy() {
}

type FfiConverterAuthMethod struct{}

var FfiConverterAuthMethodINSTANCE = FfiConverterAuthMethod{}

func (c FfiConverterAuthMethod) Lift(rb RustBufferI) AuthMethod {
	return LiftFromRustBuffer[AuthMethod](c, rb)
}

func (c FfiConverterAuthMethod) Lower(value AuthMethod) C.RustBuffer {
	return LowerIntoRustBuffer[AuthMethod](c, value)
}

func (c FfiConverterAuthMethod) LowerExternal(value AuthMethod) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[AuthMethod](c, value))
}
func (FfiConverterAuthMethod) Read(reader io.Reader) AuthMethod {
	id := readInt32(reader)
	switch id {
	case 1:
		return AuthMethodBearer{}
	case 2:
		return AuthMethodApiKeyHeader{
			FfiConverterStringINSTANCE.Read(reader),
		}
	case 3:
		return AuthMethodAzureApiKey{}
	case 4:
		return AuthMethodKeyPrefix{}
	default:
		panic(fmt.Sprintf("invalid enum value %v in FfiConverterAuthMethod.Read()", id))
	}
}

func (FfiConverterAuthMethod) Write(writer io.Writer, value AuthMethod) {
	switch variant_value := value.(type) {
	case AuthMethodBearer:
		writeInt32(writer, 1)
	case AuthMethodApiKeyHeader:
		writeInt32(writer, 2)
		FfiConverterStringINSTANCE.Write(writer, variant_value.HeaderName)
	case AuthMethodAzureApiKey:
		writeInt32(writer, 3)
	case AuthMethodKeyPrefix:
		writeInt32(writer, 4)
	default:
		_ = variant_value
		panic(fmt.Sprintf("invalid enum value `%v` in FfiConverterAuthMethod.Write", value))
	}
}

type FfiDestroyerAuthMethod struct{}

func (_ FfiDestroyerAuthMethod) Destroy(value AuthMethod) {
	value.Destroy()
}

// Per-request outcome within a [`BatchResult`].
//
// Slot `i` of [`BatchResult::responses`] corresponds to input request `i`.
// Successful slots carry the [`CompletionResponse`]; failed slots carry an
// `error_message` only (the structured `BlazenError` variant doesn't survive
// nesting inside a `uniffi::Enum` cleanly across all four target languages,
// so the message is flattened to a string here — foreign callers wanting
// typed errors should run requests individually).
type BatchItem interface {
	Destroy()
}

// The request completed and the model returned a response.
type BatchItemSuccess struct {
	Response CompletionResponse
}

func (e BatchItemSuccess) Destroy() {
	FfiDestroyerCompletionResponse{}.Destroy(e.Response)
}

// The request failed. The message mirrors the `Display` form of the
// underlying [`BlazenError`].
type BatchItemFailure struct {
	ErrorMessage string
}

func (e BatchItemFailure) Destroy() {
	FfiDestroyerString{}.Destroy(e.ErrorMessage)
}

type FfiConverterBatchItem struct{}

var FfiConverterBatchItemINSTANCE = FfiConverterBatchItem{}

func (c FfiConverterBatchItem) Lift(rb RustBufferI) BatchItem {
	return LiftFromRustBuffer[BatchItem](c, rb)
}

func (c FfiConverterBatchItem) Lower(value BatchItem) C.RustBuffer {
	return LowerIntoRustBuffer[BatchItem](c, value)
}

func (c FfiConverterBatchItem) LowerExternal(value BatchItem) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[BatchItem](c, value))
}
func (FfiConverterBatchItem) Read(reader io.Reader) BatchItem {
	id := readInt32(reader)
	switch id {
	case 1:
		return BatchItemSuccess{
			FfiConverterCompletionResponseINSTANCE.Read(reader),
		}
	case 2:
		return BatchItemFailure{
			FfiConverterStringINSTANCE.Read(reader),
		}
	default:
		panic(fmt.Sprintf("invalid enum value %v in FfiConverterBatchItem.Read()", id))
	}
}

func (FfiConverterBatchItem) Write(writer io.Writer, value BatchItem) {
	switch variant_value := value.(type) {
	case BatchItemSuccess:
		writeInt32(writer, 1)
		FfiConverterCompletionResponseINSTANCE.Write(writer, variant_value.Response)
	case BatchItemFailure:
		writeInt32(writer, 2)
		FfiConverterStringINSTANCE.Write(writer, variant_value.ErrorMessage)
	default:
		_ = variant_value
		panic(fmt.Sprintf("invalid enum value `%v` in FfiConverterBatchItem.Write", value))
	}
}

type FfiDestroyerBatchItem struct{}

func (_ FfiDestroyerBatchItem) Destroy(value BatchItem) {
	value.Destroy()
}

// Canonical error type for all Blazen UniFFI bindings.
//
// Each variant carries a `message` field with a human-readable description
// (matching the corresponding Node/Python error's `.message`). Variants with
// sub-types carry a `kind` string discriminator (e.g. `Provider.kind = "LlamaCppModelLoad"`).
type BlazenError struct {
	err error
}

// Convenience method to turn *BlazenError into error
// Avoiding treating nil pointer as non nil error interface
func (err *BlazenError) AsError() error {
	if err == nil {
		return nil
	} else {
		return err
	}
}

func (err BlazenError) Error() string {
	return fmt.Sprintf("BlazenError: %s", err.err.Error())
}

func (err BlazenError) Unwrap() error {
	return err.err
}

// Err* are used for checking error type with `errors.Is`
var ErrBlazenErrorAuth = fmt.Errorf("BlazenErrorAuth")
var ErrBlazenErrorRateLimit = fmt.Errorf("BlazenErrorRateLimit")
var ErrBlazenErrorTimeout = fmt.Errorf("BlazenErrorTimeout")
var ErrBlazenErrorValidation = fmt.Errorf("BlazenErrorValidation")
var ErrBlazenErrorContentPolicy = fmt.Errorf("BlazenErrorContentPolicy")
var ErrBlazenErrorUnsupported = fmt.Errorf("BlazenErrorUnsupported")
var ErrBlazenErrorCompute = fmt.Errorf("BlazenErrorCompute")
var ErrBlazenErrorMedia = fmt.Errorf("BlazenErrorMedia")
var ErrBlazenErrorProvider = fmt.Errorf("BlazenErrorProvider")
var ErrBlazenErrorWorkflow = fmt.Errorf("BlazenErrorWorkflow")
var ErrBlazenErrorTool = fmt.Errorf("BlazenErrorTool")
var ErrBlazenErrorPeer = fmt.Errorf("BlazenErrorPeer")
var ErrBlazenErrorPersist = fmt.Errorf("BlazenErrorPersist")
var ErrBlazenErrorPrompt = fmt.Errorf("BlazenErrorPrompt")
var ErrBlazenErrorMemory = fmt.Errorf("BlazenErrorMemory")
var ErrBlazenErrorCache = fmt.Errorf("BlazenErrorCache")
var ErrBlazenErrorCancelled = fmt.Errorf("BlazenErrorCancelled")
var ErrBlazenErrorInternal = fmt.Errorf("BlazenErrorInternal")

// Variant structs
// Authentication / credentials failure (missing API key, invalid token, etc.).
type BlazenErrorAuth struct {
	Message string
}

// Authentication / credentials failure (missing API key, invalid token, etc.).
func NewBlazenErrorAuth(
	message string,
) *BlazenError {
	return &BlazenError{err: &BlazenErrorAuth{
		Message: message}}
}

func (e BlazenErrorAuth) destroy() {
	FfiDestroyerString{}.Destroy(e.Message)
}

func (err BlazenErrorAuth) Error() string {
	return fmt.Sprint("Auth",
		": ",

		"Message=",
		err.Message,
	)
}

func (self BlazenErrorAuth) Is(target error) bool {
	return target == ErrBlazenErrorAuth
}

// Rate limit exceeded. `retry_after_ms` is set when the provider returned a
// Retry-After hint.
type BlazenErrorRateLimit struct {
	Message      string
	RetryAfterMs *uint64
}

// Rate limit exceeded. `retry_after_ms` is set when the provider returned a
// Retry-After hint.
func NewBlazenErrorRateLimit(
	message string,
	retryAfterMs *uint64,
) *BlazenError {
	return &BlazenError{err: &BlazenErrorRateLimit{
		Message:      message,
		RetryAfterMs: retryAfterMs}}
}

func (e BlazenErrorRateLimit) destroy() {
	FfiDestroyerString{}.Destroy(e.Message)
	FfiDestroyerOptionalUint64{}.Destroy(e.RetryAfterMs)
}

func (err BlazenErrorRateLimit) Error() string {
	return fmt.Sprint("RateLimit",
		": ",

		"Message=",
		err.Message,
		", ",
		"RetryAfterMs=",
		err.RetryAfterMs,
	)
}

func (self BlazenErrorRateLimit) Is(target error) bool {
	return target == ErrBlazenErrorRateLimit
}

// Operation timed out before the provider responded.
type BlazenErrorTimeout struct {
	Message   string
	ElapsedMs uint64
}

// Operation timed out before the provider responded.
func NewBlazenErrorTimeout(
	message string,
	elapsedMs uint64,
) *BlazenError {
	return &BlazenError{err: &BlazenErrorTimeout{
		Message:   message,
		ElapsedMs: elapsedMs}}
}

func (e BlazenErrorTimeout) destroy() {
	FfiDestroyerString{}.Destroy(e.Message)
	FfiDestroyerUint64{}.Destroy(e.ElapsedMs)
}

func (err BlazenErrorTimeout) Error() string {
	return fmt.Sprint("Timeout",
		": ",

		"Message=",
		err.Message,
		", ",
		"ElapsedMs=",
		err.ElapsedMs,
	)
}

func (self BlazenErrorTimeout) Is(target error) bool {
	return target == ErrBlazenErrorTimeout
}

// Input validation failed (bad schema, missing required field, etc.).
type BlazenErrorValidation struct {
	Message string
}

// Input validation failed (bad schema, missing required field, etc.).
func NewBlazenErrorValidation(
	message string,
) *BlazenError {
	return &BlazenError{err: &BlazenErrorValidation{
		Message: message}}
}

func (e BlazenErrorValidation) destroy() {
	FfiDestroyerString{}.Destroy(e.Message)
}

func (err BlazenErrorValidation) Error() string {
	return fmt.Sprint("Validation",
		": ",

		"Message=",
		err.Message,
	)
}

func (self BlazenErrorValidation) Is(target error) bool {
	return target == ErrBlazenErrorValidation
}

// Content policy violation (provider refused due to safety filters).
type BlazenErrorContentPolicy struct {
	Message string
}

// Content policy violation (provider refused due to safety filters).
func NewBlazenErrorContentPolicy(
	message string,
) *BlazenError {
	return &BlazenError{err: &BlazenErrorContentPolicy{
		Message: message}}
}

func (e BlazenErrorContentPolicy) destroy() {
	FfiDestroyerString{}.Destroy(e.Message)
}

func (err BlazenErrorContentPolicy) Error() string {
	return fmt.Sprint("ContentPolicy",
		": ",

		"Message=",
		err.Message,
	)
}

func (self BlazenErrorContentPolicy) Is(target error) bool {
	return target == ErrBlazenErrorContentPolicy
}

// Operation unsupported on this platform / build / provider.
type BlazenErrorUnsupported struct {
	Message string
}

// Operation unsupported on this platform / build / provider.
func NewBlazenErrorUnsupported(
	message string,
) *BlazenError {
	return &BlazenError{err: &BlazenErrorUnsupported{
		Message: message}}
}

func (e BlazenErrorUnsupported) destroy() {
	FfiDestroyerString{}.Destroy(e.Message)
}

func (err BlazenErrorUnsupported) Error() string {
	return fmt.Sprint("Unsupported",
		": ",

		"Message=",
		err.Message,
	)
}

func (self BlazenErrorUnsupported) Is(target error) bool {
	return target == ErrBlazenErrorUnsupported
}

// Compute error (CPU/GPU/accelerator failure, OOM, etc.).
type BlazenErrorCompute struct {
	Message string
}

// Compute error (CPU/GPU/accelerator failure, OOM, etc.).
func NewBlazenErrorCompute(
	message string,
) *BlazenError {
	return &BlazenError{err: &BlazenErrorCompute{
		Message: message}}
}

func (e BlazenErrorCompute) destroy() {
	FfiDestroyerString{}.Destroy(e.Message)
}

func (err BlazenErrorCompute) Error() string {
	return fmt.Sprint("Compute",
		": ",

		"Message=",
		err.Message,
	)
}

func (self BlazenErrorCompute) Is(target error) bool {
	return target == ErrBlazenErrorCompute
}

// Media-handling error (decode, encode, transcoding).
type BlazenErrorMedia struct {
	Message string
}

// Media-handling error (decode, encode, transcoding).
func NewBlazenErrorMedia(
	message string,
) *BlazenError {
	return &BlazenError{err: &BlazenErrorMedia{
		Message: message}}
}

func (e BlazenErrorMedia) destroy() {
	FfiDestroyerString{}.Destroy(e.Message)
}

func (err BlazenErrorMedia) Error() string {
	return fmt.Sprint("Media",
		": ",

		"Message=",
		err.Message,
	)
}

func (self BlazenErrorMedia) Is(target error) bool {
	return target == ErrBlazenErrorMedia
}

// Provider / backend error. `kind` identifies the specific backend and failure
// mode, mirroring the Node binding's `[ProviderError]` sentinel JSON shape.
// Examples of `kind`: `"LlamaCppModelLoad"`, `"DiffusionGeneration"`,
// `"CandleEmbedInference"`, `"OpenAIHttp"`, `"AnthropicHttp"`.
type BlazenErrorProvider struct {
	Kind         string
	Message      string
	Provider     *string
	Status       *uint32
	Endpoint     *string
	RequestId    *string
	Detail       *string
	RetryAfterMs *uint64
}

// Provider / backend error. `kind` identifies the specific backend and failure
// mode, mirroring the Node binding's `[ProviderError]` sentinel JSON shape.
// Examples of `kind`: `"LlamaCppModelLoad"`, `"DiffusionGeneration"`,
// `"CandleEmbedInference"`, `"OpenAIHttp"`, `"AnthropicHttp"`.
func NewBlazenErrorProvider(
	kind string,
	message string,
	provider *string,
	status *uint32,
	endpoint *string,
	requestId *string,
	detail *string,
	retryAfterMs *uint64,
) *BlazenError {
	return &BlazenError{err: &BlazenErrorProvider{
		Kind:         kind,
		Message:      message,
		Provider:     provider,
		Status:       status,
		Endpoint:     endpoint,
		RequestId:    requestId,
		Detail:       detail,
		RetryAfterMs: retryAfterMs}}
}

func (e BlazenErrorProvider) destroy() {
	FfiDestroyerString{}.Destroy(e.Kind)
	FfiDestroyerString{}.Destroy(e.Message)
	FfiDestroyerOptionalString{}.Destroy(e.Provider)
	FfiDestroyerOptionalUint32{}.Destroy(e.Status)
	FfiDestroyerOptionalString{}.Destroy(e.Endpoint)
	FfiDestroyerOptionalString{}.Destroy(e.RequestId)
	FfiDestroyerOptionalString{}.Destroy(e.Detail)
	FfiDestroyerOptionalUint64{}.Destroy(e.RetryAfterMs)
}

func (err BlazenErrorProvider) Error() string {
	return fmt.Sprint("Provider",
		": ",

		"Kind=",
		err.Kind,
		", ",
		"Message=",
		err.Message,
		", ",
		"Provider=",
		err.Provider,
		", ",
		"Status=",
		err.Status,
		", ",
		"Endpoint=",
		err.Endpoint,
		", ",
		"RequestId=",
		err.RequestId,
		", ",
		"Detail=",
		err.Detail,
		", ",
		"RetryAfterMs=",
		err.RetryAfterMs,
	)
}

func (self BlazenErrorProvider) Is(target error) bool {
	return target == ErrBlazenErrorProvider
}

// Workflow execution error (step panic, deadlock, missing context, etc.).
type BlazenErrorWorkflow struct {
	Message string
}

// Workflow execution error (step panic, deadlock, missing context, etc.).
func NewBlazenErrorWorkflow(
	message string,
) *BlazenError {
	return &BlazenError{err: &BlazenErrorWorkflow{
		Message: message}}
}

func (e BlazenErrorWorkflow) destroy() {
	FfiDestroyerString{}.Destroy(e.Message)
}

func (err BlazenErrorWorkflow) Error() string {
	return fmt.Sprint("Workflow",
		": ",

		"Message=",
		err.Message,
	)
}

func (self BlazenErrorWorkflow) Is(target error) bool {
	return target == ErrBlazenErrorWorkflow
}

// Tool / function-call error during LLM agent execution.
type BlazenErrorTool struct {
	Message string
}

// Tool / function-call error during LLM agent execution.
func NewBlazenErrorTool(
	message string,
) *BlazenError {
	return &BlazenError{err: &BlazenErrorTool{
		Message: message}}
}

func (e BlazenErrorTool) destroy() {
	FfiDestroyerString{}.Destroy(e.Message)
}

func (err BlazenErrorTool) Error() string {
	return fmt.Sprint("Tool",
		": ",

		"Message=",
		err.Message,
	)
}

func (self BlazenErrorTool) Is(target error) bool {
	return target == ErrBlazenErrorTool
}

// Distributed peer-to-peer error. `kind` is one of: `"Encode"`, `"Transport"`,
// `"EnvelopeVersion"`, `"Workflow"`, `"Tls"`, `"UnknownStep"`.
type BlazenErrorPeer struct {
	Kind    string
	Message string
}

// Distributed peer-to-peer error. `kind` is one of: `"Encode"`, `"Transport"`,
// `"EnvelopeVersion"`, `"Workflow"`, `"Tls"`, `"UnknownStep"`.
func NewBlazenErrorPeer(
	kind string,
	message string,
) *BlazenError {
	return &BlazenError{err: &BlazenErrorPeer{
		Kind:    kind,
		Message: message}}
}

func (e BlazenErrorPeer) destroy() {
	FfiDestroyerString{}.Destroy(e.Kind)
	FfiDestroyerString{}.Destroy(e.Message)
}

func (err BlazenErrorPeer) Error() string {
	return fmt.Sprint("Peer",
		": ",

		"Kind=",
		err.Kind,
		", ",
		"Message=",
		err.Message,
	)
}

func (self BlazenErrorPeer) Is(target error) bool {
	return target == ErrBlazenErrorPeer
}

// Persistence layer error (redb / valkey checkpoint store).
type BlazenErrorPersist struct {
	Message string
}

// Persistence layer error (redb / valkey checkpoint store).
func NewBlazenErrorPersist(
	message string,
) *BlazenError {
	return &BlazenError{err: &BlazenErrorPersist{
		Message: message}}
}

func (e BlazenErrorPersist) destroy() {
	FfiDestroyerString{}.Destroy(e.Message)
}

func (err BlazenErrorPersist) Error() string {
	return fmt.Sprint("Persist",
		": ",

		"Message=",
		err.Message,
	)
}

func (self BlazenErrorPersist) Is(target error) bool {
	return target == ErrBlazenErrorPersist
}

// Prompt template error. `kind`: `"MissingVariable"`, `"NotFound"`, `"VersionNotFound"`,
// `"Io"`, `"Yaml"`, `"Json"`, `"Validation"`.
type BlazenErrorPrompt struct {
	Kind    string
	Message string
}

// Prompt template error. `kind`: `"MissingVariable"`, `"NotFound"`, `"VersionNotFound"`,
// `"Io"`, `"Yaml"`, `"Json"`, `"Validation"`.
func NewBlazenErrorPrompt(
	kind string,
	message string,
) *BlazenError {
	return &BlazenError{err: &BlazenErrorPrompt{
		Kind:    kind,
		Message: message}}
}

func (e BlazenErrorPrompt) destroy() {
	FfiDestroyerString{}.Destroy(e.Kind)
	FfiDestroyerString{}.Destroy(e.Message)
}

func (err BlazenErrorPrompt) Error() string {
	return fmt.Sprint("Prompt",
		": ",

		"Kind=",
		err.Kind,
		", ",
		"Message=",
		err.Message,
	)
}

func (self BlazenErrorPrompt) Is(target error) bool {
	return target == ErrBlazenErrorPrompt
}

// Memory subsystem error. `kind`: `"NoEmbedder"`, `"Elid"`, `"Embedding"`,
// `"NotFound"`, `"Serialization"`, `"Io"`, `"Backend"`.
type BlazenErrorMemory struct {
	Kind    string
	Message string
}

// Memory subsystem error. `kind`: `"NoEmbedder"`, `"Elid"`, `"Embedding"`,
// `"NotFound"`, `"Serialization"`, `"Io"`, `"Backend"`.
func NewBlazenErrorMemory(
	kind string,
	message string,
) *BlazenError {
	return &BlazenError{err: &BlazenErrorMemory{
		Kind:    kind,
		Message: message}}
}

func (e BlazenErrorMemory) destroy() {
	FfiDestroyerString{}.Destroy(e.Kind)
	FfiDestroyerString{}.Destroy(e.Message)
}

func (err BlazenErrorMemory) Error() string {
	return fmt.Sprint("Memory",
		": ",

		"Kind=",
		err.Kind,
		", ",
		"Message=",
		err.Message,
	)
}

func (self BlazenErrorMemory) Is(target error) bool {
	return target == ErrBlazenErrorMemory
}

// Model-cache / download error. `kind`: `"Download"`, `"CacheDir"`, `"Io"`.
type BlazenErrorCache struct {
	Kind    string
	Message string
}

// Model-cache / download error. `kind`: `"Download"`, `"CacheDir"`, `"Io"`.
func NewBlazenErrorCache(
	kind string,
	message string,
) *BlazenError {
	return &BlazenError{err: &BlazenErrorCache{
		Kind:    kind,
		Message: message}}
}

func (e BlazenErrorCache) destroy() {
	FfiDestroyerString{}.Destroy(e.Kind)
	FfiDestroyerString{}.Destroy(e.Message)
}

func (err BlazenErrorCache) Error() string {
	return fmt.Sprint("Cache",
		": ",

		"Kind=",
		err.Kind,
		", ",
		"Message=",
		err.Message,
	)
}

func (self BlazenErrorCache) Is(target error) bool {
	return target == ErrBlazenErrorCache
}

// Operation was cancelled (e.g. via a foreign-language `context.Context`
// or `Task.cancel()` request). Mapped to `context.Canceled` /
// `Task.CancellationError` / `Kotlin CancellationException` on the foreign side.
type BlazenErrorCancelled struct {
}

// Operation was cancelled (e.g. via a foreign-language `context.Context`
// or `Task.cancel()` request). Mapped to `context.Canceled` /
// `Task.CancellationError` / `Kotlin CancellationException` on the foreign side.
func NewBlazenErrorCancelled() *BlazenError {
	return &BlazenError{err: &BlazenErrorCancelled{}}
}

func (e BlazenErrorCancelled) destroy() {
}

func (err BlazenErrorCancelled) Error() string {
	return fmt.Sprint("Cancelled")
}

func (self BlazenErrorCancelled) Is(target error) bool {
	return target == ErrBlazenErrorCancelled
}

// Fallback for errors that don't fit any other variant — should be rare;
// new errors should usually get their own variant or a `kind` extension.
type BlazenErrorInternal struct {
	Message string
}

// Fallback for errors that don't fit any other variant — should be rare;
// new errors should usually get their own variant or a `kind` extension.
func NewBlazenErrorInternal(
	message string,
) *BlazenError {
	return &BlazenError{err: &BlazenErrorInternal{
		Message: message}}
}

func (e BlazenErrorInternal) destroy() {
	FfiDestroyerString{}.Destroy(e.Message)
}

func (err BlazenErrorInternal) Error() string {
	return fmt.Sprint("Internal",
		": ",

		"Message=",
		err.Message,
	)
}

func (self BlazenErrorInternal) Is(target error) bool {
	return target == ErrBlazenErrorInternal
}

type FfiConverterBlazenError struct{}

var FfiConverterBlazenErrorINSTANCE = FfiConverterBlazenError{}

func (c FfiConverterBlazenError) Lift(eb RustBufferI) *BlazenError {
	return LiftFromRustBuffer[*BlazenError](c, eb)
}

func (c FfiConverterBlazenError) Lower(value *BlazenError) C.RustBuffer {
	return LowerIntoRustBuffer[*BlazenError](c, value)
}

func (c FfiConverterBlazenError) LowerExternal(value *BlazenError) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[*BlazenError](c, value))
}

func (c FfiConverterBlazenError) Read(reader io.Reader) *BlazenError {
	errorID := readUint32(reader)

	switch errorID {
	case 1:
		return &BlazenError{&BlazenErrorAuth{
			Message: FfiConverterStringINSTANCE.Read(reader),
		}}
	case 2:
		return &BlazenError{&BlazenErrorRateLimit{
			Message:      FfiConverterStringINSTANCE.Read(reader),
			RetryAfterMs: FfiConverterOptionalUint64INSTANCE.Read(reader),
		}}
	case 3:
		return &BlazenError{&BlazenErrorTimeout{
			Message:   FfiConverterStringINSTANCE.Read(reader),
			ElapsedMs: FfiConverterUint64INSTANCE.Read(reader),
		}}
	case 4:
		return &BlazenError{&BlazenErrorValidation{
			Message: FfiConverterStringINSTANCE.Read(reader),
		}}
	case 5:
		return &BlazenError{&BlazenErrorContentPolicy{
			Message: FfiConverterStringINSTANCE.Read(reader),
		}}
	case 6:
		return &BlazenError{&BlazenErrorUnsupported{
			Message: FfiConverterStringINSTANCE.Read(reader),
		}}
	case 7:
		return &BlazenError{&BlazenErrorCompute{
			Message: FfiConverterStringINSTANCE.Read(reader),
		}}
	case 8:
		return &BlazenError{&BlazenErrorMedia{
			Message: FfiConverterStringINSTANCE.Read(reader),
		}}
	case 9:
		return &BlazenError{&BlazenErrorProvider{
			Kind:         FfiConverterStringINSTANCE.Read(reader),
			Message:      FfiConverterStringINSTANCE.Read(reader),
			Provider:     FfiConverterOptionalStringINSTANCE.Read(reader),
			Status:       FfiConverterOptionalUint32INSTANCE.Read(reader),
			Endpoint:     FfiConverterOptionalStringINSTANCE.Read(reader),
			RequestId:    FfiConverterOptionalStringINSTANCE.Read(reader),
			Detail:       FfiConverterOptionalStringINSTANCE.Read(reader),
			RetryAfterMs: FfiConverterOptionalUint64INSTANCE.Read(reader),
		}}
	case 10:
		return &BlazenError{&BlazenErrorWorkflow{
			Message: FfiConverterStringINSTANCE.Read(reader),
		}}
	case 11:
		return &BlazenError{&BlazenErrorTool{
			Message: FfiConverterStringINSTANCE.Read(reader),
		}}
	case 12:
		return &BlazenError{&BlazenErrorPeer{
			Kind:    FfiConverterStringINSTANCE.Read(reader),
			Message: FfiConverterStringINSTANCE.Read(reader),
		}}
	case 13:
		return &BlazenError{&BlazenErrorPersist{
			Message: FfiConverterStringINSTANCE.Read(reader),
		}}
	case 14:
		return &BlazenError{&BlazenErrorPrompt{
			Kind:    FfiConverterStringINSTANCE.Read(reader),
			Message: FfiConverterStringINSTANCE.Read(reader),
		}}
	case 15:
		return &BlazenError{&BlazenErrorMemory{
			Kind:    FfiConverterStringINSTANCE.Read(reader),
			Message: FfiConverterStringINSTANCE.Read(reader),
		}}
	case 16:
		return &BlazenError{&BlazenErrorCache{
			Kind:    FfiConverterStringINSTANCE.Read(reader),
			Message: FfiConverterStringINSTANCE.Read(reader),
		}}
	case 17:
		return &BlazenError{&BlazenErrorCancelled{}}
	case 18:
		return &BlazenError{&BlazenErrorInternal{
			Message: FfiConverterStringINSTANCE.Read(reader),
		}}
	default:
		panic(fmt.Sprintf("Unknown error code %d in FfiConverterBlazenError.Read()", errorID))
	}
}

func (c FfiConverterBlazenError) Write(writer io.Writer, value *BlazenError) {
	switch variantValue := value.err.(type) {
	case *BlazenErrorAuth:
		writeInt32(writer, 1)
		FfiConverterStringINSTANCE.Write(writer, variantValue.Message)
	case *BlazenErrorRateLimit:
		writeInt32(writer, 2)
		FfiConverterStringINSTANCE.Write(writer, variantValue.Message)
		FfiConverterOptionalUint64INSTANCE.Write(writer, variantValue.RetryAfterMs)
	case *BlazenErrorTimeout:
		writeInt32(writer, 3)
		FfiConverterStringINSTANCE.Write(writer, variantValue.Message)
		FfiConverterUint64INSTANCE.Write(writer, variantValue.ElapsedMs)
	case *BlazenErrorValidation:
		writeInt32(writer, 4)
		FfiConverterStringINSTANCE.Write(writer, variantValue.Message)
	case *BlazenErrorContentPolicy:
		writeInt32(writer, 5)
		FfiConverterStringINSTANCE.Write(writer, variantValue.Message)
	case *BlazenErrorUnsupported:
		writeInt32(writer, 6)
		FfiConverterStringINSTANCE.Write(writer, variantValue.Message)
	case *BlazenErrorCompute:
		writeInt32(writer, 7)
		FfiConverterStringINSTANCE.Write(writer, variantValue.Message)
	case *BlazenErrorMedia:
		writeInt32(writer, 8)
		FfiConverterStringINSTANCE.Write(writer, variantValue.Message)
	case *BlazenErrorProvider:
		writeInt32(writer, 9)
		FfiConverterStringINSTANCE.Write(writer, variantValue.Kind)
		FfiConverterStringINSTANCE.Write(writer, variantValue.Message)
		FfiConverterOptionalStringINSTANCE.Write(writer, variantValue.Provider)
		FfiConverterOptionalUint32INSTANCE.Write(writer, variantValue.Status)
		FfiConverterOptionalStringINSTANCE.Write(writer, variantValue.Endpoint)
		FfiConverterOptionalStringINSTANCE.Write(writer, variantValue.RequestId)
		FfiConverterOptionalStringINSTANCE.Write(writer, variantValue.Detail)
		FfiConverterOptionalUint64INSTANCE.Write(writer, variantValue.RetryAfterMs)
	case *BlazenErrorWorkflow:
		writeInt32(writer, 10)
		FfiConverterStringINSTANCE.Write(writer, variantValue.Message)
	case *BlazenErrorTool:
		writeInt32(writer, 11)
		FfiConverterStringINSTANCE.Write(writer, variantValue.Message)
	case *BlazenErrorPeer:
		writeInt32(writer, 12)
		FfiConverterStringINSTANCE.Write(writer, variantValue.Kind)
		FfiConverterStringINSTANCE.Write(writer, variantValue.Message)
	case *BlazenErrorPersist:
		writeInt32(writer, 13)
		FfiConverterStringINSTANCE.Write(writer, variantValue.Message)
	case *BlazenErrorPrompt:
		writeInt32(writer, 14)
		FfiConverterStringINSTANCE.Write(writer, variantValue.Kind)
		FfiConverterStringINSTANCE.Write(writer, variantValue.Message)
	case *BlazenErrorMemory:
		writeInt32(writer, 15)
		FfiConverterStringINSTANCE.Write(writer, variantValue.Kind)
		FfiConverterStringINSTANCE.Write(writer, variantValue.Message)
	case *BlazenErrorCache:
		writeInt32(writer, 16)
		FfiConverterStringINSTANCE.Write(writer, variantValue.Kind)
		FfiConverterStringINSTANCE.Write(writer, variantValue.Message)
	case *BlazenErrorCancelled:
		writeInt32(writer, 17)
	case *BlazenErrorInternal:
		writeInt32(writer, 18)
		FfiConverterStringINSTANCE.Write(writer, variantValue.Message)
	default:
		_ = variantValue
		panic(fmt.Sprintf("invalid error value `%v` in FfiConverterBlazenError.Write", value))
	}
}

type FfiDestroyerBlazenError struct{}

func (_ FfiDestroyerBlazenError) Destroy(value *BlazenError) {
	switch variantValue := value.err.(type) {
	case BlazenErrorAuth:
		variantValue.destroy()
	case BlazenErrorRateLimit:
		variantValue.destroy()
	case BlazenErrorTimeout:
		variantValue.destroy()
	case BlazenErrorValidation:
		variantValue.destroy()
	case BlazenErrorContentPolicy:
		variantValue.destroy()
	case BlazenErrorUnsupported:
		variantValue.destroy()
	case BlazenErrorCompute:
		variantValue.destroy()
	case BlazenErrorMedia:
		variantValue.destroy()
	case BlazenErrorProvider:
		variantValue.destroy()
	case BlazenErrorWorkflow:
		variantValue.destroy()
	case BlazenErrorTool:
		variantValue.destroy()
	case BlazenErrorPeer:
		variantValue.destroy()
	case BlazenErrorPersist:
		variantValue.destroy()
	case BlazenErrorPrompt:
		variantValue.destroy()
	case BlazenErrorMemory:
		variantValue.destroy()
	case BlazenErrorCache:
		variantValue.destroy()
	case BlazenErrorCancelled:
		variantValue.destroy()
	case BlazenErrorInternal:
		variantValue.destroy()
	default:
		_ = variantValue
		panic(fmt.Sprintf("invalid error value `%v` in FfiDestroyerBlazenError.Destroy", value))
	}
}

// What a [`StepHandler`] returns: zero, one, or many events to publish.
type StepOutput interface {
	Destroy()
}

// Step performed work but produced no event.
type StepOutputNone struct {
}

func (e StepOutputNone) Destroy() {
}

// Step produced exactly one event (the common case).
type StepOutputSingle struct {
	Event Event
}

func (e StepOutputSingle) Destroy() {
	FfiDestroyerEvent{}.Destroy(e.Event)
}

// Step fans out — produced multiple events at once.
type StepOutputMultiple struct {
	Events []Event
}

func (e StepOutputMultiple) Destroy() {
	FfiDestroyerSequenceEvent{}.Destroy(e.Events)
}

type FfiConverterStepOutput struct{}

var FfiConverterStepOutputINSTANCE = FfiConverterStepOutput{}

func (c FfiConverterStepOutput) Lift(rb RustBufferI) StepOutput {
	return LiftFromRustBuffer[StepOutput](c, rb)
}

func (c FfiConverterStepOutput) Lower(value StepOutput) C.RustBuffer {
	return LowerIntoRustBuffer[StepOutput](c, value)
}

func (c FfiConverterStepOutput) LowerExternal(value StepOutput) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[StepOutput](c, value))
}
func (FfiConverterStepOutput) Read(reader io.Reader) StepOutput {
	id := readInt32(reader)
	switch id {
	case 1:
		return StepOutputNone{}
	case 2:
		return StepOutputSingle{
			FfiConverterEventINSTANCE.Read(reader),
		}
	case 3:
		return StepOutputMultiple{
			FfiConverterSequenceEventINSTANCE.Read(reader),
		}
	default:
		panic(fmt.Sprintf("invalid enum value %v in FfiConverterStepOutput.Read()", id))
	}
}

func (FfiConverterStepOutput) Write(writer io.Writer, value StepOutput) {
	switch variant_value := value.(type) {
	case StepOutputNone:
		writeInt32(writer, 1)
	case StepOutputSingle:
		writeInt32(writer, 2)
		FfiConverterEventINSTANCE.Write(writer, variant_value.Event)
	case StepOutputMultiple:
		writeInt32(writer, 3)
		FfiConverterSequenceEventINSTANCE.Write(writer, variant_value.Events)
	default:
		_ = variant_value
		panic(fmt.Sprintf("invalid enum value `%v` in FfiConverterStepOutput.Write", value))
	}
}

type FfiDestroyerStepOutput struct{}

func (_ FfiDestroyerStepOutput) Destroy(value StepOutput) {
	value.Destroy()
}

type FfiConverterOptionalUint16 struct{}

var FfiConverterOptionalUint16INSTANCE = FfiConverterOptionalUint16{}

func (c FfiConverterOptionalUint16) Lift(rb RustBufferI) *uint16 {
	return LiftFromRustBuffer[*uint16](c, rb)
}

func (_ FfiConverterOptionalUint16) Read(reader io.Reader) *uint16 {
	if readInt8(reader) == 0 {
		return nil
	}
	temp := FfiConverterUint16INSTANCE.Read(reader)
	return &temp
}

func (c FfiConverterOptionalUint16) Lower(value *uint16) C.RustBuffer {
	return LowerIntoRustBuffer[*uint16](c, value)
}

func (c FfiConverterOptionalUint16) LowerExternal(value *uint16) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[*uint16](c, value))
}

func (_ FfiConverterOptionalUint16) Write(writer io.Writer, value *uint16) {
	if value == nil {
		writeInt8(writer, 0)
	} else {
		writeInt8(writer, 1)
		FfiConverterUint16INSTANCE.Write(writer, *value)
	}
}

type FfiDestroyerOptionalUint16 struct{}

func (_ FfiDestroyerOptionalUint16) Destroy(value *uint16) {
	if value != nil {
		FfiDestroyerUint16{}.Destroy(*value)
	}
}

type FfiConverterOptionalUint32 struct{}

var FfiConverterOptionalUint32INSTANCE = FfiConverterOptionalUint32{}

func (c FfiConverterOptionalUint32) Lift(rb RustBufferI) *uint32 {
	return LiftFromRustBuffer[*uint32](c, rb)
}

func (_ FfiConverterOptionalUint32) Read(reader io.Reader) *uint32 {
	if readInt8(reader) == 0 {
		return nil
	}
	temp := FfiConverterUint32INSTANCE.Read(reader)
	return &temp
}

func (c FfiConverterOptionalUint32) Lower(value *uint32) C.RustBuffer {
	return LowerIntoRustBuffer[*uint32](c, value)
}

func (c FfiConverterOptionalUint32) LowerExternal(value *uint32) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[*uint32](c, value))
}

func (_ FfiConverterOptionalUint32) Write(writer io.Writer, value *uint32) {
	if value == nil {
		writeInt8(writer, 0)
	} else {
		writeInt8(writer, 1)
		FfiConverterUint32INSTANCE.Write(writer, *value)
	}
}

type FfiDestroyerOptionalUint32 struct{}

func (_ FfiDestroyerOptionalUint32) Destroy(value *uint32) {
	if value != nil {
		FfiDestroyerUint32{}.Destroy(*value)
	}
}

type FfiConverterOptionalUint64 struct{}

var FfiConverterOptionalUint64INSTANCE = FfiConverterOptionalUint64{}

func (c FfiConverterOptionalUint64) Lift(rb RustBufferI) *uint64 {
	return LiftFromRustBuffer[*uint64](c, rb)
}

func (_ FfiConverterOptionalUint64) Read(reader io.Reader) *uint64 {
	if readInt8(reader) == 0 {
		return nil
	}
	temp := FfiConverterUint64INSTANCE.Read(reader)
	return &temp
}

func (c FfiConverterOptionalUint64) Lower(value *uint64) C.RustBuffer {
	return LowerIntoRustBuffer[*uint64](c, value)
}

func (c FfiConverterOptionalUint64) LowerExternal(value *uint64) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[*uint64](c, value))
}

func (_ FfiConverterOptionalUint64) Write(writer io.Writer, value *uint64) {
	if value == nil {
		writeInt8(writer, 0)
	} else {
		writeInt8(writer, 1)
		FfiConverterUint64INSTANCE.Write(writer, *value)
	}
}

type FfiDestroyerOptionalUint64 struct{}

func (_ FfiDestroyerOptionalUint64) Destroy(value *uint64) {
	if value != nil {
		FfiDestroyerUint64{}.Destroy(*value)
	}
}

type FfiConverterOptionalFloat32 struct{}

var FfiConverterOptionalFloat32INSTANCE = FfiConverterOptionalFloat32{}

func (c FfiConverterOptionalFloat32) Lift(rb RustBufferI) *float32 {
	return LiftFromRustBuffer[*float32](c, rb)
}

func (_ FfiConverterOptionalFloat32) Read(reader io.Reader) *float32 {
	if readInt8(reader) == 0 {
		return nil
	}
	temp := FfiConverterFloat32INSTANCE.Read(reader)
	return &temp
}

func (c FfiConverterOptionalFloat32) Lower(value *float32) C.RustBuffer {
	return LowerIntoRustBuffer[*float32](c, value)
}

func (c FfiConverterOptionalFloat32) LowerExternal(value *float32) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[*float32](c, value))
}

func (_ FfiConverterOptionalFloat32) Write(writer io.Writer, value *float32) {
	if value == nil {
		writeInt8(writer, 0)
	} else {
		writeInt8(writer, 1)
		FfiConverterFloat32INSTANCE.Write(writer, *value)
	}
}

type FfiDestroyerOptionalFloat32 struct{}

func (_ FfiDestroyerOptionalFloat32) Destroy(value *float32) {
	if value != nil {
		FfiDestroyerFloat32{}.Destroy(*value)
	}
}

type FfiConverterOptionalFloat64 struct{}

var FfiConverterOptionalFloat64INSTANCE = FfiConverterOptionalFloat64{}

func (c FfiConverterOptionalFloat64) Lift(rb RustBufferI) *float64 {
	return LiftFromRustBuffer[*float64](c, rb)
}

func (_ FfiConverterOptionalFloat64) Read(reader io.Reader) *float64 {
	if readInt8(reader) == 0 {
		return nil
	}
	temp := FfiConverterFloat64INSTANCE.Read(reader)
	return &temp
}

func (c FfiConverterOptionalFloat64) Lower(value *float64) C.RustBuffer {
	return LowerIntoRustBuffer[*float64](c, value)
}

func (c FfiConverterOptionalFloat64) LowerExternal(value *float64) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[*float64](c, value))
}

func (_ FfiConverterOptionalFloat64) Write(writer io.Writer, value *float64) {
	if value == nil {
		writeInt8(writer, 0)
	} else {
		writeInt8(writer, 1)
		FfiConverterFloat64INSTANCE.Write(writer, *value)
	}
}

type FfiDestroyerOptionalFloat64 struct{}

func (_ FfiDestroyerOptionalFloat64) Destroy(value *float64) {
	if value != nil {
		FfiDestroyerFloat64{}.Destroy(*value)
	}
}

type FfiConverterOptionalBool struct{}

var FfiConverterOptionalBoolINSTANCE = FfiConverterOptionalBool{}

func (c FfiConverterOptionalBool) Lift(rb RustBufferI) *bool {
	return LiftFromRustBuffer[*bool](c, rb)
}

func (_ FfiConverterOptionalBool) Read(reader io.Reader) *bool {
	if readInt8(reader) == 0 {
		return nil
	}
	temp := FfiConverterBoolINSTANCE.Read(reader)
	return &temp
}

func (c FfiConverterOptionalBool) Lower(value *bool) C.RustBuffer {
	return LowerIntoRustBuffer[*bool](c, value)
}

func (c FfiConverterOptionalBool) LowerExternal(value *bool) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[*bool](c, value))
}

func (_ FfiConverterOptionalBool) Write(writer io.Writer, value *bool) {
	if value == nil {
		writeInt8(writer, 0)
	} else {
		writeInt8(writer, 1)
		FfiConverterBoolINSTANCE.Write(writer, *value)
	}
}

type FfiDestroyerOptionalBool struct{}

func (_ FfiDestroyerOptionalBool) Destroy(value *bool) {
	if value != nil {
		FfiDestroyerBool{}.Destroy(*value)
	}
}

type FfiConverterOptionalString struct{}

var FfiConverterOptionalStringINSTANCE = FfiConverterOptionalString{}

func (c FfiConverterOptionalString) Lift(rb RustBufferI) *string {
	return LiftFromRustBuffer[*string](c, rb)
}

func (_ FfiConverterOptionalString) Read(reader io.Reader) *string {
	if readInt8(reader) == 0 {
		return nil
	}
	temp := FfiConverterStringINSTANCE.Read(reader)
	return &temp
}

func (c FfiConverterOptionalString) Lower(value *string) C.RustBuffer {
	return LowerIntoRustBuffer[*string](c, value)
}

func (c FfiConverterOptionalString) LowerExternal(value *string) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[*string](c, value))
}

func (_ FfiConverterOptionalString) Write(writer io.Writer, value *string) {
	if value == nil {
		writeInt8(writer, 0)
	} else {
		writeInt8(writer, 1)
		FfiConverterStringINSTANCE.Write(writer, *value)
	}
}

type FfiDestroyerOptionalString struct{}

func (_ FfiDestroyerOptionalString) Destroy(value *string) {
	if value != nil {
		FfiDestroyerString{}.Destroy(*value)
	}
}

type FfiConverterOptionalBaseProviderDefaults struct{}

var FfiConverterOptionalBaseProviderDefaultsINSTANCE = FfiConverterOptionalBaseProviderDefaults{}

func (c FfiConverterOptionalBaseProviderDefaults) Lift(rb RustBufferI) *BaseProviderDefaults {
	return LiftFromRustBuffer[*BaseProviderDefaults](c, rb)
}

func (_ FfiConverterOptionalBaseProviderDefaults) Read(reader io.Reader) *BaseProviderDefaults {
	if readInt8(reader) == 0 {
		return nil
	}
	temp := FfiConverterBaseProviderDefaultsINSTANCE.Read(reader)
	return &temp
}

func (c FfiConverterOptionalBaseProviderDefaults) Lower(value *BaseProviderDefaults) C.RustBuffer {
	return LowerIntoRustBuffer[*BaseProviderDefaults](c, value)
}

func (c FfiConverterOptionalBaseProviderDefaults) LowerExternal(value *BaseProviderDefaults) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[*BaseProviderDefaults](c, value))
}

func (_ FfiConverterOptionalBaseProviderDefaults) Write(writer io.Writer, value *BaseProviderDefaults) {
	if value == nil {
		writeInt8(writer, 0)
	} else {
		writeInt8(writer, 1)
		FfiConverterBaseProviderDefaultsINSTANCE.Write(writer, *value)
	}
}

type FfiDestroyerOptionalBaseProviderDefaults struct{}

func (_ FfiDestroyerOptionalBaseProviderDefaults) Destroy(value *BaseProviderDefaults) {
	if value != nil {
		FfiDestroyerBaseProviderDefaults{}.Destroy(*value)
	}
}

type FfiConverterOptionalTokenUsage struct{}

var FfiConverterOptionalTokenUsageINSTANCE = FfiConverterOptionalTokenUsage{}

func (c FfiConverterOptionalTokenUsage) Lift(rb RustBufferI) *TokenUsage {
	return LiftFromRustBuffer[*TokenUsage](c, rb)
}

func (_ FfiConverterOptionalTokenUsage) Read(reader io.Reader) *TokenUsage {
	if readInt8(reader) == 0 {
		return nil
	}
	temp := FfiConverterTokenUsageINSTANCE.Read(reader)
	return &temp
}

func (c FfiConverterOptionalTokenUsage) Lower(value *TokenUsage) C.RustBuffer {
	return LowerIntoRustBuffer[*TokenUsage](c, value)
}

func (c FfiConverterOptionalTokenUsage) LowerExternal(value *TokenUsage) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[*TokenUsage](c, value))
}

func (_ FfiConverterOptionalTokenUsage) Write(writer io.Writer, value *TokenUsage) {
	if value == nil {
		writeInt8(writer, 0)
	} else {
		writeInt8(writer, 1)
		FfiConverterTokenUsageINSTANCE.Write(writer, *value)
	}
}

type FfiDestroyerOptionalTokenUsage struct{}

func (_ FfiDestroyerOptionalTokenUsage) Destroy(value *TokenUsage) {
	if value != nil {
		FfiDestroyerTokenUsage{}.Destroy(*value)
	}
}

type FfiConverterOptionalWorkflowCheckpoint struct{}

var FfiConverterOptionalWorkflowCheckpointINSTANCE = FfiConverterOptionalWorkflowCheckpoint{}

func (c FfiConverterOptionalWorkflowCheckpoint) Lift(rb RustBufferI) *WorkflowCheckpoint {
	return LiftFromRustBuffer[*WorkflowCheckpoint](c, rb)
}

func (_ FfiConverterOptionalWorkflowCheckpoint) Read(reader io.Reader) *WorkflowCheckpoint {
	if readInt8(reader) == 0 {
		return nil
	}
	temp := FfiConverterWorkflowCheckpointINSTANCE.Read(reader)
	return &temp
}

func (c FfiConverterOptionalWorkflowCheckpoint) Lower(value *WorkflowCheckpoint) C.RustBuffer {
	return LowerIntoRustBuffer[*WorkflowCheckpoint](c, value)
}

func (c FfiConverterOptionalWorkflowCheckpoint) LowerExternal(value *WorkflowCheckpoint) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[*WorkflowCheckpoint](c, value))
}

func (_ FfiConverterOptionalWorkflowCheckpoint) Write(writer io.Writer, value *WorkflowCheckpoint) {
	if value == nil {
		writeInt8(writer, 0)
	} else {
		writeInt8(writer, 1)
		FfiConverterWorkflowCheckpointINSTANCE.Write(writer, *value)
	}
}

type FfiDestroyerOptionalWorkflowCheckpoint struct{}

func (_ FfiDestroyerOptionalWorkflowCheckpoint) Destroy(value *WorkflowCheckpoint) {
	if value != nil {
		FfiDestroyerWorkflowCheckpoint{}.Destroy(*value)
	}
}

type FfiConverterSequenceFloat64 struct{}

var FfiConverterSequenceFloat64INSTANCE = FfiConverterSequenceFloat64{}

func (c FfiConverterSequenceFloat64) Lift(rb RustBufferI) []float64 {
	return LiftFromRustBuffer[[]float64](c, rb)
}

func (c FfiConverterSequenceFloat64) Read(reader io.Reader) []float64 {
	length := readInt32(reader)
	if length == 0 {
		return nil
	}
	result := make([]float64, 0, length)
	for i := int32(0); i < length; i++ {
		result = append(result, FfiConverterFloat64INSTANCE.Read(reader))
	}
	return result
}

func (c FfiConverterSequenceFloat64) Lower(value []float64) C.RustBuffer {
	return LowerIntoRustBuffer[[]float64](c, value)
}

func (c FfiConverterSequenceFloat64) LowerExternal(value []float64) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[[]float64](c, value))
}

func (c FfiConverterSequenceFloat64) Write(writer io.Writer, value []float64) {
	if len(value) > math.MaxInt32 {
		panic("[]float64 is too large to fit into Int32")
	}

	writeInt32(writer, int32(len(value)))
	for _, item := range value {
		FfiConverterFloat64INSTANCE.Write(writer, item)
	}
}

type FfiDestroyerSequenceFloat64 struct{}

func (FfiDestroyerSequenceFloat64) Destroy(sequence []float64) {
	for _, value := range sequence {
		FfiDestroyerFloat64{}.Destroy(value)
	}
}

type FfiConverterSequenceString struct{}

var FfiConverterSequenceStringINSTANCE = FfiConverterSequenceString{}

func (c FfiConverterSequenceString) Lift(rb RustBufferI) []string {
	return LiftFromRustBuffer[[]string](c, rb)
}

func (c FfiConverterSequenceString) Read(reader io.Reader) []string {
	length := readInt32(reader)
	if length == 0 {
		return nil
	}
	result := make([]string, 0, length)
	for i := int32(0); i < length; i++ {
		result = append(result, FfiConverterStringINSTANCE.Read(reader))
	}
	return result
}

func (c FfiConverterSequenceString) Lower(value []string) C.RustBuffer {
	return LowerIntoRustBuffer[[]string](c, value)
}

func (c FfiConverterSequenceString) LowerExternal(value []string) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[[]string](c, value))
}

func (c FfiConverterSequenceString) Write(writer io.Writer, value []string) {
	if len(value) > math.MaxInt32 {
		panic("[]string is too large to fit into Int32")
	}

	writeInt32(writer, int32(len(value)))
	for _, item := range value {
		FfiConverterStringINSTANCE.Write(writer, item)
	}
}

type FfiDestroyerSequenceString struct{}

func (FfiDestroyerSequenceString) Destroy(sequence []string) {
	for _, value := range sequence {
		FfiDestroyerString{}.Destroy(value)
	}
}

type FfiConverterSequenceWorkflow struct{}

var FfiConverterSequenceWorkflowINSTANCE = FfiConverterSequenceWorkflow{}

func (c FfiConverterSequenceWorkflow) Lift(rb RustBufferI) []*Workflow {
	return LiftFromRustBuffer[[]*Workflow](c, rb)
}

func (c FfiConverterSequenceWorkflow) Read(reader io.Reader) []*Workflow {
	length := readInt32(reader)
	if length == 0 {
		return nil
	}
	result := make([]*Workflow, 0, length)
	for i := int32(0); i < length; i++ {
		result = append(result, FfiConverterWorkflowINSTANCE.Read(reader))
	}
	return result
}

func (c FfiConverterSequenceWorkflow) Lower(value []*Workflow) C.RustBuffer {
	return LowerIntoRustBuffer[[]*Workflow](c, value)
}

func (c FfiConverterSequenceWorkflow) LowerExternal(value []*Workflow) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[[]*Workflow](c, value))
}

func (c FfiConverterSequenceWorkflow) Write(writer io.Writer, value []*Workflow) {
	if len(value) > math.MaxInt32 {
		panic("[]*Workflow is too large to fit into Int32")
	}

	writeInt32(writer, int32(len(value)))
	for _, item := range value {
		FfiConverterWorkflowINSTANCE.Write(writer, item)
	}
}

type FfiDestroyerSequenceWorkflow struct{}

func (FfiDestroyerSequenceWorkflow) Destroy(sequence []*Workflow) {
	for _, value := range sequence {
		FfiDestroyerWorkflow{}.Destroy(value)
	}
}

type FfiConverterSequenceChatMessage struct{}

var FfiConverterSequenceChatMessageINSTANCE = FfiConverterSequenceChatMessage{}

func (c FfiConverterSequenceChatMessage) Lift(rb RustBufferI) []ChatMessage {
	return LiftFromRustBuffer[[]ChatMessage](c, rb)
}

func (c FfiConverterSequenceChatMessage) Read(reader io.Reader) []ChatMessage {
	length := readInt32(reader)
	if length == 0 {
		return nil
	}
	result := make([]ChatMessage, 0, length)
	for i := int32(0); i < length; i++ {
		result = append(result, FfiConverterChatMessageINSTANCE.Read(reader))
	}
	return result
}

func (c FfiConverterSequenceChatMessage) Lower(value []ChatMessage) C.RustBuffer {
	return LowerIntoRustBuffer[[]ChatMessage](c, value)
}

func (c FfiConverterSequenceChatMessage) LowerExternal(value []ChatMessage) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[[]ChatMessage](c, value))
}

func (c FfiConverterSequenceChatMessage) Write(writer io.Writer, value []ChatMessage) {
	if len(value) > math.MaxInt32 {
		panic("[]ChatMessage is too large to fit into Int32")
	}

	writeInt32(writer, int32(len(value)))
	for _, item := range value {
		FfiConverterChatMessageINSTANCE.Write(writer, item)
	}
}

type FfiDestroyerSequenceChatMessage struct{}

func (FfiDestroyerSequenceChatMessage) Destroy(sequence []ChatMessage) {
	for _, value := range sequence {
		FfiDestroyerChatMessage{}.Destroy(value)
	}
}

type FfiConverterSequenceCompletionRequest struct{}

var FfiConverterSequenceCompletionRequestINSTANCE = FfiConverterSequenceCompletionRequest{}

func (c FfiConverterSequenceCompletionRequest) Lift(rb RustBufferI) []CompletionRequest {
	return LiftFromRustBuffer[[]CompletionRequest](c, rb)
}

func (c FfiConverterSequenceCompletionRequest) Read(reader io.Reader) []CompletionRequest {
	length := readInt32(reader)
	if length == 0 {
		return nil
	}
	result := make([]CompletionRequest, 0, length)
	for i := int32(0); i < length; i++ {
		result = append(result, FfiConverterCompletionRequestINSTANCE.Read(reader))
	}
	return result
}

func (c FfiConverterSequenceCompletionRequest) Lower(value []CompletionRequest) C.RustBuffer {
	return LowerIntoRustBuffer[[]CompletionRequest](c, value)
}

func (c FfiConverterSequenceCompletionRequest) LowerExternal(value []CompletionRequest) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[[]CompletionRequest](c, value))
}

func (c FfiConverterSequenceCompletionRequest) Write(writer io.Writer, value []CompletionRequest) {
	if len(value) > math.MaxInt32 {
		panic("[]CompletionRequest is too large to fit into Int32")
	}

	writeInt32(writer, int32(len(value)))
	for _, item := range value {
		FfiConverterCompletionRequestINSTANCE.Write(writer, item)
	}
}

type FfiDestroyerSequenceCompletionRequest struct{}

func (FfiDestroyerSequenceCompletionRequest) Destroy(sequence []CompletionRequest) {
	for _, value := range sequence {
		FfiDestroyerCompletionRequest{}.Destroy(value)
	}
}

type FfiConverterSequenceEvent struct{}

var FfiConverterSequenceEventINSTANCE = FfiConverterSequenceEvent{}

func (c FfiConverterSequenceEvent) Lift(rb RustBufferI) []Event {
	return LiftFromRustBuffer[[]Event](c, rb)
}

func (c FfiConverterSequenceEvent) Read(reader io.Reader) []Event {
	length := readInt32(reader)
	if length == 0 {
		return nil
	}
	result := make([]Event, 0, length)
	for i := int32(0); i < length; i++ {
		result = append(result, FfiConverterEventINSTANCE.Read(reader))
	}
	return result
}

func (c FfiConverterSequenceEvent) Lower(value []Event) C.RustBuffer {
	return LowerIntoRustBuffer[[]Event](c, value)
}

func (c FfiConverterSequenceEvent) LowerExternal(value []Event) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[[]Event](c, value))
}

func (c FfiConverterSequenceEvent) Write(writer io.Writer, value []Event) {
	if len(value) > math.MaxInt32 {
		panic("[]Event is too large to fit into Int32")
	}

	writeInt32(writer, int32(len(value)))
	for _, item := range value {
		FfiConverterEventINSTANCE.Write(writer, item)
	}
}

type FfiDestroyerSequenceEvent struct{}

func (FfiDestroyerSequenceEvent) Destroy(sequence []Event) {
	for _, value := range sequence {
		FfiDestroyerEvent{}.Destroy(value)
	}
}

type FfiConverterSequenceGenerated3DModel struct{}

var FfiConverterSequenceGenerated3DModelINSTANCE = FfiConverterSequenceGenerated3DModel{}

func (c FfiConverterSequenceGenerated3DModel) Lift(rb RustBufferI) []Generated3DModel {
	return LiftFromRustBuffer[[]Generated3DModel](c, rb)
}

func (c FfiConverterSequenceGenerated3DModel) Read(reader io.Reader) []Generated3DModel {
	length := readInt32(reader)
	if length == 0 {
		return nil
	}
	result := make([]Generated3DModel, 0, length)
	for i := int32(0); i < length; i++ {
		result = append(result, FfiConverterGenerated3DModelINSTANCE.Read(reader))
	}
	return result
}

func (c FfiConverterSequenceGenerated3DModel) Lower(value []Generated3DModel) C.RustBuffer {
	return LowerIntoRustBuffer[[]Generated3DModel](c, value)
}

func (c FfiConverterSequenceGenerated3DModel) LowerExternal(value []Generated3DModel) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[[]Generated3DModel](c, value))
}

func (c FfiConverterSequenceGenerated3DModel) Write(writer io.Writer, value []Generated3DModel) {
	if len(value) > math.MaxInt32 {
		panic("[]Generated3DModel is too large to fit into Int32")
	}

	writeInt32(writer, int32(len(value)))
	for _, item := range value {
		FfiConverterGenerated3DModelINSTANCE.Write(writer, item)
	}
}

type FfiDestroyerSequenceGenerated3DModel struct{}

func (FfiDestroyerSequenceGenerated3DModel) Destroy(sequence []Generated3DModel) {
	for _, value := range sequence {
		FfiDestroyerGenerated3DModel{}.Destroy(value)
	}
}

type FfiConverterSequenceGeneratedAudio struct{}

var FfiConverterSequenceGeneratedAudioINSTANCE = FfiConverterSequenceGeneratedAudio{}

func (c FfiConverterSequenceGeneratedAudio) Lift(rb RustBufferI) []GeneratedAudio {
	return LiftFromRustBuffer[[]GeneratedAudio](c, rb)
}

func (c FfiConverterSequenceGeneratedAudio) Read(reader io.Reader) []GeneratedAudio {
	length := readInt32(reader)
	if length == 0 {
		return nil
	}
	result := make([]GeneratedAudio, 0, length)
	for i := int32(0); i < length; i++ {
		result = append(result, FfiConverterGeneratedAudioINSTANCE.Read(reader))
	}
	return result
}

func (c FfiConverterSequenceGeneratedAudio) Lower(value []GeneratedAudio) C.RustBuffer {
	return LowerIntoRustBuffer[[]GeneratedAudio](c, value)
}

func (c FfiConverterSequenceGeneratedAudio) LowerExternal(value []GeneratedAudio) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[[]GeneratedAudio](c, value))
}

func (c FfiConverterSequenceGeneratedAudio) Write(writer io.Writer, value []GeneratedAudio) {
	if len(value) > math.MaxInt32 {
		panic("[]GeneratedAudio is too large to fit into Int32")
	}

	writeInt32(writer, int32(len(value)))
	for _, item := range value {
		FfiConverterGeneratedAudioINSTANCE.Write(writer, item)
	}
}

type FfiDestroyerSequenceGeneratedAudio struct{}

func (FfiDestroyerSequenceGeneratedAudio) Destroy(sequence []GeneratedAudio) {
	for _, value := range sequence {
		FfiDestroyerGeneratedAudio{}.Destroy(value)
	}
}

type FfiConverterSequenceGeneratedImage struct{}

var FfiConverterSequenceGeneratedImageINSTANCE = FfiConverterSequenceGeneratedImage{}

func (c FfiConverterSequenceGeneratedImage) Lift(rb RustBufferI) []GeneratedImage {
	return LiftFromRustBuffer[[]GeneratedImage](c, rb)
}

func (c FfiConverterSequenceGeneratedImage) Read(reader io.Reader) []GeneratedImage {
	length := readInt32(reader)
	if length == 0 {
		return nil
	}
	result := make([]GeneratedImage, 0, length)
	for i := int32(0); i < length; i++ {
		result = append(result, FfiConverterGeneratedImageINSTANCE.Read(reader))
	}
	return result
}

func (c FfiConverterSequenceGeneratedImage) Lower(value []GeneratedImage) C.RustBuffer {
	return LowerIntoRustBuffer[[]GeneratedImage](c, value)
}

func (c FfiConverterSequenceGeneratedImage) LowerExternal(value []GeneratedImage) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[[]GeneratedImage](c, value))
}

func (c FfiConverterSequenceGeneratedImage) Write(writer io.Writer, value []GeneratedImage) {
	if len(value) > math.MaxInt32 {
		panic("[]GeneratedImage is too large to fit into Int32")
	}

	writeInt32(writer, int32(len(value)))
	for _, item := range value {
		FfiConverterGeneratedImageINSTANCE.Write(writer, item)
	}
}

type FfiDestroyerSequenceGeneratedImage struct{}

func (FfiDestroyerSequenceGeneratedImage) Destroy(sequence []GeneratedImage) {
	for _, value := range sequence {
		FfiDestroyerGeneratedImage{}.Destroy(value)
	}
}

type FfiConverterSequenceGeneratedVideo struct{}

var FfiConverterSequenceGeneratedVideoINSTANCE = FfiConverterSequenceGeneratedVideo{}

func (c FfiConverterSequenceGeneratedVideo) Lift(rb RustBufferI) []GeneratedVideo {
	return LiftFromRustBuffer[[]GeneratedVideo](c, rb)
}

func (c FfiConverterSequenceGeneratedVideo) Read(reader io.Reader) []GeneratedVideo {
	length := readInt32(reader)
	if length == 0 {
		return nil
	}
	result := make([]GeneratedVideo, 0, length)
	for i := int32(0); i < length; i++ {
		result = append(result, FfiConverterGeneratedVideoINSTANCE.Read(reader))
	}
	return result
}

func (c FfiConverterSequenceGeneratedVideo) Lower(value []GeneratedVideo) C.RustBuffer {
	return LowerIntoRustBuffer[[]GeneratedVideo](c, value)
}

func (c FfiConverterSequenceGeneratedVideo) LowerExternal(value []GeneratedVideo) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[[]GeneratedVideo](c, value))
}

func (c FfiConverterSequenceGeneratedVideo) Write(writer io.Writer, value []GeneratedVideo) {
	if len(value) > math.MaxInt32 {
		panic("[]GeneratedVideo is too large to fit into Int32")
	}

	writeInt32(writer, int32(len(value)))
	for _, item := range value {
		FfiConverterGeneratedVideoINSTANCE.Write(writer, item)
	}
}

type FfiDestroyerSequenceGeneratedVideo struct{}

func (FfiDestroyerSequenceGeneratedVideo) Destroy(sequence []GeneratedVideo) {
	for _, value := range sequence {
		FfiDestroyerGeneratedVideo{}.Destroy(value)
	}
}

type FfiConverterSequenceKeyValue struct{}

var FfiConverterSequenceKeyValueINSTANCE = FfiConverterSequenceKeyValue{}

func (c FfiConverterSequenceKeyValue) Lift(rb RustBufferI) []KeyValue {
	return LiftFromRustBuffer[[]KeyValue](c, rb)
}

func (c FfiConverterSequenceKeyValue) Read(reader io.Reader) []KeyValue {
	length := readInt32(reader)
	if length == 0 {
		return nil
	}
	result := make([]KeyValue, 0, length)
	for i := int32(0); i < length; i++ {
		result = append(result, FfiConverterKeyValueINSTANCE.Read(reader))
	}
	return result
}

func (c FfiConverterSequenceKeyValue) Lower(value []KeyValue) C.RustBuffer {
	return LowerIntoRustBuffer[[]KeyValue](c, value)
}

func (c FfiConverterSequenceKeyValue) LowerExternal(value []KeyValue) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[[]KeyValue](c, value))
}

func (c FfiConverterSequenceKeyValue) Write(writer io.Writer, value []KeyValue) {
	if len(value) > math.MaxInt32 {
		panic("[]KeyValue is too large to fit into Int32")
	}

	writeInt32(writer, int32(len(value)))
	for _, item := range value {
		FfiConverterKeyValueINSTANCE.Write(writer, item)
	}
}

type FfiDestroyerSequenceKeyValue struct{}

func (FfiDestroyerSequenceKeyValue) Destroy(sequence []KeyValue) {
	for _, value := range sequence {
		FfiDestroyerKeyValue{}.Destroy(value)
	}
}

type FfiConverterSequenceMedia struct{}

var FfiConverterSequenceMediaINSTANCE = FfiConverterSequenceMedia{}

func (c FfiConverterSequenceMedia) Lift(rb RustBufferI) []Media {
	return LiftFromRustBuffer[[]Media](c, rb)
}

func (c FfiConverterSequenceMedia) Read(reader io.Reader) []Media {
	length := readInt32(reader)
	if length == 0 {
		return nil
	}
	result := make([]Media, 0, length)
	for i := int32(0); i < length; i++ {
		result = append(result, FfiConverterMediaINSTANCE.Read(reader))
	}
	return result
}

func (c FfiConverterSequenceMedia) Lower(value []Media) C.RustBuffer {
	return LowerIntoRustBuffer[[]Media](c, value)
}

func (c FfiConverterSequenceMedia) LowerExternal(value []Media) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[[]Media](c, value))
}

func (c FfiConverterSequenceMedia) Write(writer io.Writer, value []Media) {
	if len(value) > math.MaxInt32 {
		panic("[]Media is too large to fit into Int32")
	}

	writeInt32(writer, int32(len(value)))
	for _, item := range value {
		FfiConverterMediaINSTANCE.Write(writer, item)
	}
}

type FfiDestroyerSequenceMedia struct{}

func (FfiDestroyerSequenceMedia) Destroy(sequence []Media) {
	for _, value := range sequence {
		FfiDestroyerMedia{}.Destroy(value)
	}
}

type FfiConverterSequencePersistedEvent struct{}

var FfiConverterSequencePersistedEventINSTANCE = FfiConverterSequencePersistedEvent{}

func (c FfiConverterSequencePersistedEvent) Lift(rb RustBufferI) []PersistedEvent {
	return LiftFromRustBuffer[[]PersistedEvent](c, rb)
}

func (c FfiConverterSequencePersistedEvent) Read(reader io.Reader) []PersistedEvent {
	length := readInt32(reader)
	if length == 0 {
		return nil
	}
	result := make([]PersistedEvent, 0, length)
	for i := int32(0); i < length; i++ {
		result = append(result, FfiConverterPersistedEventINSTANCE.Read(reader))
	}
	return result
}

func (c FfiConverterSequencePersistedEvent) Lower(value []PersistedEvent) C.RustBuffer {
	return LowerIntoRustBuffer[[]PersistedEvent](c, value)
}

func (c FfiConverterSequencePersistedEvent) LowerExternal(value []PersistedEvent) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[[]PersistedEvent](c, value))
}

func (c FfiConverterSequencePersistedEvent) Write(writer io.Writer, value []PersistedEvent) {
	if len(value) > math.MaxInt32 {
		panic("[]PersistedEvent is too large to fit into Int32")
	}

	writeInt32(writer, int32(len(value)))
	for _, item := range value {
		FfiConverterPersistedEventINSTANCE.Write(writer, item)
	}
}

type FfiDestroyerSequencePersistedEvent struct{}

func (FfiDestroyerSequencePersistedEvent) Destroy(sequence []PersistedEvent) {
	for _, value := range sequence {
		FfiDestroyerPersistedEvent{}.Destroy(value)
	}
}

type FfiConverterSequenceTool struct{}

var FfiConverterSequenceToolINSTANCE = FfiConverterSequenceTool{}

func (c FfiConverterSequenceTool) Lift(rb RustBufferI) []Tool {
	return LiftFromRustBuffer[[]Tool](c, rb)
}

func (c FfiConverterSequenceTool) Read(reader io.Reader) []Tool {
	length := readInt32(reader)
	if length == 0 {
		return nil
	}
	result := make([]Tool, 0, length)
	for i := int32(0); i < length; i++ {
		result = append(result, FfiConverterToolINSTANCE.Read(reader))
	}
	return result
}

func (c FfiConverterSequenceTool) Lower(value []Tool) C.RustBuffer {
	return LowerIntoRustBuffer[[]Tool](c, value)
}

func (c FfiConverterSequenceTool) LowerExternal(value []Tool) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[[]Tool](c, value))
}

func (c FfiConverterSequenceTool) Write(writer io.Writer, value []Tool) {
	if len(value) > math.MaxInt32 {
		panic("[]Tool is too large to fit into Int32")
	}

	writeInt32(writer, int32(len(value)))
	for _, item := range value {
		FfiConverterToolINSTANCE.Write(writer, item)
	}
}

type FfiDestroyerSequenceTool struct{}

func (FfiDestroyerSequenceTool) Destroy(sequence []Tool) {
	for _, value := range sequence {
		FfiDestroyerTool{}.Destroy(value)
	}
}

type FfiConverterSequenceToolCall struct{}

var FfiConverterSequenceToolCallINSTANCE = FfiConverterSequenceToolCall{}

func (c FfiConverterSequenceToolCall) Lift(rb RustBufferI) []ToolCall {
	return LiftFromRustBuffer[[]ToolCall](c, rb)
}

func (c FfiConverterSequenceToolCall) Read(reader io.Reader) []ToolCall {
	length := readInt32(reader)
	if length == 0 {
		return nil
	}
	result := make([]ToolCall, 0, length)
	for i := int32(0); i < length; i++ {
		result = append(result, FfiConverterToolCallINSTANCE.Read(reader))
	}
	return result
}

func (c FfiConverterSequenceToolCall) Lower(value []ToolCall) C.RustBuffer {
	return LowerIntoRustBuffer[[]ToolCall](c, value)
}

func (c FfiConverterSequenceToolCall) LowerExternal(value []ToolCall) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[[]ToolCall](c, value))
}

func (c FfiConverterSequenceToolCall) Write(writer io.Writer, value []ToolCall) {
	if len(value) > math.MaxInt32 {
		panic("[]ToolCall is too large to fit into Int32")
	}

	writeInt32(writer, int32(len(value)))
	for _, item := range value {
		FfiConverterToolCallINSTANCE.Write(writer, item)
	}
}

type FfiDestroyerSequenceToolCall struct{}

func (FfiDestroyerSequenceToolCall) Destroy(sequence []ToolCall) {
	for _, value := range sequence {
		FfiDestroyerToolCall{}.Destroy(value)
	}
}

type FfiConverterSequenceTranscriptionSegment struct{}

var FfiConverterSequenceTranscriptionSegmentINSTANCE = FfiConverterSequenceTranscriptionSegment{}

func (c FfiConverterSequenceTranscriptionSegment) Lift(rb RustBufferI) []TranscriptionSegment {
	return LiftFromRustBuffer[[]TranscriptionSegment](c, rb)
}

func (c FfiConverterSequenceTranscriptionSegment) Read(reader io.Reader) []TranscriptionSegment {
	length := readInt32(reader)
	if length == 0 {
		return nil
	}
	result := make([]TranscriptionSegment, 0, length)
	for i := int32(0); i < length; i++ {
		result = append(result, FfiConverterTranscriptionSegmentINSTANCE.Read(reader))
	}
	return result
}

func (c FfiConverterSequenceTranscriptionSegment) Lower(value []TranscriptionSegment) C.RustBuffer {
	return LowerIntoRustBuffer[[]TranscriptionSegment](c, value)
}

func (c FfiConverterSequenceTranscriptionSegment) LowerExternal(value []TranscriptionSegment) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[[]TranscriptionSegment](c, value))
}

func (c FfiConverterSequenceTranscriptionSegment) Write(writer io.Writer, value []TranscriptionSegment) {
	if len(value) > math.MaxInt32 {
		panic("[]TranscriptionSegment is too large to fit into Int32")
	}

	writeInt32(writer, int32(len(value)))
	for _, item := range value {
		FfiConverterTranscriptionSegmentINSTANCE.Write(writer, item)
	}
}

type FfiDestroyerSequenceTranscriptionSegment struct{}

func (FfiDestroyerSequenceTranscriptionSegment) Destroy(sequence []TranscriptionSegment) {
	for _, value := range sequence {
		FfiDestroyerTranscriptionSegment{}.Destroy(value)
	}
}

type FfiConverterSequenceVoiceHandle struct{}

var FfiConverterSequenceVoiceHandleINSTANCE = FfiConverterSequenceVoiceHandle{}

func (c FfiConverterSequenceVoiceHandle) Lift(rb RustBufferI) []VoiceHandle {
	return LiftFromRustBuffer[[]VoiceHandle](c, rb)
}

func (c FfiConverterSequenceVoiceHandle) Read(reader io.Reader) []VoiceHandle {
	length := readInt32(reader)
	if length == 0 {
		return nil
	}
	result := make([]VoiceHandle, 0, length)
	for i := int32(0); i < length; i++ {
		result = append(result, FfiConverterVoiceHandleINSTANCE.Read(reader))
	}
	return result
}

func (c FfiConverterSequenceVoiceHandle) Lower(value []VoiceHandle) C.RustBuffer {
	return LowerIntoRustBuffer[[]VoiceHandle](c, value)
}

func (c FfiConverterSequenceVoiceHandle) LowerExternal(value []VoiceHandle) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[[]VoiceHandle](c, value))
}

func (c FfiConverterSequenceVoiceHandle) Write(writer io.Writer, value []VoiceHandle) {
	if len(value) > math.MaxInt32 {
		panic("[]VoiceHandle is too large to fit into Int32")
	}

	writeInt32(writer, int32(len(value)))
	for _, item := range value {
		FfiConverterVoiceHandleINSTANCE.Write(writer, item)
	}
}

type FfiDestroyerSequenceVoiceHandle struct{}

func (FfiDestroyerSequenceVoiceHandle) Destroy(sequence []VoiceHandle) {
	for _, value := range sequence {
		FfiDestroyerVoiceHandle{}.Destroy(value)
	}
}

type FfiConverterSequenceWorkflowCheckpoint struct{}

var FfiConverterSequenceWorkflowCheckpointINSTANCE = FfiConverterSequenceWorkflowCheckpoint{}

func (c FfiConverterSequenceWorkflowCheckpoint) Lift(rb RustBufferI) []WorkflowCheckpoint {
	return LiftFromRustBuffer[[]WorkflowCheckpoint](c, rb)
}

func (c FfiConverterSequenceWorkflowCheckpoint) Read(reader io.Reader) []WorkflowCheckpoint {
	length := readInt32(reader)
	if length == 0 {
		return nil
	}
	result := make([]WorkflowCheckpoint, 0, length)
	for i := int32(0); i < length; i++ {
		result = append(result, FfiConverterWorkflowCheckpointINSTANCE.Read(reader))
	}
	return result
}

func (c FfiConverterSequenceWorkflowCheckpoint) Lower(value []WorkflowCheckpoint) C.RustBuffer {
	return LowerIntoRustBuffer[[]WorkflowCheckpoint](c, value)
}

func (c FfiConverterSequenceWorkflowCheckpoint) LowerExternal(value []WorkflowCheckpoint) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[[]WorkflowCheckpoint](c, value))
}

func (c FfiConverterSequenceWorkflowCheckpoint) Write(writer io.Writer, value []WorkflowCheckpoint) {
	if len(value) > math.MaxInt32 {
		panic("[]WorkflowCheckpoint is too large to fit into Int32")
	}

	writeInt32(writer, int32(len(value)))
	for _, item := range value {
		FfiConverterWorkflowCheckpointINSTANCE.Write(writer, item)
	}
}

type FfiDestroyerSequenceWorkflowCheckpoint struct{}

func (FfiDestroyerSequenceWorkflowCheckpoint) Destroy(sequence []WorkflowCheckpoint) {
	for _, value := range sequence {
		FfiDestroyerWorkflowCheckpoint{}.Destroy(value)
	}
}

type FfiConverterSequenceWorkflowHistoryEntry struct{}

var FfiConverterSequenceWorkflowHistoryEntryINSTANCE = FfiConverterSequenceWorkflowHistoryEntry{}

func (c FfiConverterSequenceWorkflowHistoryEntry) Lift(rb RustBufferI) []WorkflowHistoryEntry {
	return LiftFromRustBuffer[[]WorkflowHistoryEntry](c, rb)
}

func (c FfiConverterSequenceWorkflowHistoryEntry) Read(reader io.Reader) []WorkflowHistoryEntry {
	length := readInt32(reader)
	if length == 0 {
		return nil
	}
	result := make([]WorkflowHistoryEntry, 0, length)
	for i := int32(0); i < length; i++ {
		result = append(result, FfiConverterWorkflowHistoryEntryINSTANCE.Read(reader))
	}
	return result
}

func (c FfiConverterSequenceWorkflowHistoryEntry) Lower(value []WorkflowHistoryEntry) C.RustBuffer {
	return LowerIntoRustBuffer[[]WorkflowHistoryEntry](c, value)
}

func (c FfiConverterSequenceWorkflowHistoryEntry) LowerExternal(value []WorkflowHistoryEntry) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[[]WorkflowHistoryEntry](c, value))
}

func (c FfiConverterSequenceWorkflowHistoryEntry) Write(writer io.Writer, value []WorkflowHistoryEntry) {
	if len(value) > math.MaxInt32 {
		panic("[]WorkflowHistoryEntry is too large to fit into Int32")
	}

	writeInt32(writer, int32(len(value)))
	for _, item := range value {
		FfiConverterWorkflowHistoryEntryINSTANCE.Write(writer, item)
	}
}

type FfiDestroyerSequenceWorkflowHistoryEntry struct{}

func (FfiDestroyerSequenceWorkflowHistoryEntry) Destroy(sequence []WorkflowHistoryEntry) {
	for _, value := range sequence {
		FfiDestroyerWorkflowHistoryEntry{}.Destroy(value)
	}
}

type FfiConverterSequenceBatchItem struct{}

var FfiConverterSequenceBatchItemINSTANCE = FfiConverterSequenceBatchItem{}

func (c FfiConverterSequenceBatchItem) Lift(rb RustBufferI) []BatchItem {
	return LiftFromRustBuffer[[]BatchItem](c, rb)
}

func (c FfiConverterSequenceBatchItem) Read(reader io.Reader) []BatchItem {
	length := readInt32(reader)
	if length == 0 {
		return nil
	}
	result := make([]BatchItem, 0, length)
	for i := int32(0); i < length; i++ {
		result = append(result, FfiConverterBatchItemINSTANCE.Read(reader))
	}
	return result
}

func (c FfiConverterSequenceBatchItem) Lower(value []BatchItem) C.RustBuffer {
	return LowerIntoRustBuffer[[]BatchItem](c, value)
}

func (c FfiConverterSequenceBatchItem) LowerExternal(value []BatchItem) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[[]BatchItem](c, value))
}

func (c FfiConverterSequenceBatchItem) Write(writer io.Writer, value []BatchItem) {
	if len(value) > math.MaxInt32 {
		panic("[]BatchItem is too large to fit into Int32")
	}

	writeInt32(writer, int32(len(value)))
	for _, item := range value {
		FfiConverterBatchItemINSTANCE.Write(writer, item)
	}
}

type FfiDestroyerSequenceBatchItem struct{}

func (FfiDestroyerSequenceBatchItem) Destroy(sequence []BatchItem) {
	for _, value := range sequence {
		FfiDestroyerBatchItem{}.Destroy(value)
	}
}

type FfiConverterSequenceSequenceFloat64 struct{}

var FfiConverterSequenceSequenceFloat64INSTANCE = FfiConverterSequenceSequenceFloat64{}

func (c FfiConverterSequenceSequenceFloat64) Lift(rb RustBufferI) [][]float64 {
	return LiftFromRustBuffer[[][]float64](c, rb)
}

func (c FfiConverterSequenceSequenceFloat64) Read(reader io.Reader) [][]float64 {
	length := readInt32(reader)
	if length == 0 {
		return nil
	}
	result := make([][]float64, 0, length)
	for i := int32(0); i < length; i++ {
		result = append(result, FfiConverterSequenceFloat64INSTANCE.Read(reader))
	}
	return result
}

func (c FfiConverterSequenceSequenceFloat64) Lower(value [][]float64) C.RustBuffer {
	return LowerIntoRustBuffer[[][]float64](c, value)
}

func (c FfiConverterSequenceSequenceFloat64) LowerExternal(value [][]float64) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[[][]float64](c, value))
}

func (c FfiConverterSequenceSequenceFloat64) Write(writer io.Writer, value [][]float64) {
	if len(value) > math.MaxInt32 {
		panic("[][]float64 is too large to fit into Int32")
	}

	writeInt32(writer, int32(len(value)))
	for _, item := range value {
		FfiConverterSequenceFloat64INSTANCE.Write(writer, item)
	}
}

type FfiDestroyerSequenceSequenceFloat64 struct{}

func (FfiDestroyerSequenceSequenceFloat64) Destroy(sequence [][]float64) {
	for _, value := range sequence {
		FfiDestroyerSequenceFloat64{}.Destroy(value)
	}
}

const (
	uniffiRustFuturePollReady      int8 = 0
	uniffiRustFuturePollMaybeReady int8 = 1
)

type rustFuturePollFunc func(C.uint64_t, C.UniffiRustFutureContinuationCallback, C.uint64_t)
type rustFutureCompleteFunc[T any] func(C.uint64_t, *C.RustCallStatus) T
type rustFutureFreeFunc func(C.uint64_t)

//export blazen_uniffiFutureContinuationCallback
func blazen_uniffiFutureContinuationCallback(data C.uint64_t, pollResult C.int8_t) {
	h := cgo.Handle(uintptr(data))
	waiter := h.Value().(chan int8)
	waiter <- int8(pollResult)
}

func uniffiRustCallAsync[E any, T any, F any](
	errConverter BufReader[E],
	completeFunc rustFutureCompleteFunc[F],
	liftFunc func(F) T,
	rustFuture C.uint64_t,
	pollFunc rustFuturePollFunc,
	freeFunc rustFutureFreeFunc,
) (T, E) {
	defer freeFunc(rustFuture)

	pollResult := int8(-1)
	waiter := make(chan int8, 1)

	chanHandle := cgo.NewHandle(waiter)
	defer chanHandle.Delete()

	for pollResult != uniffiRustFuturePollReady {
		pollFunc(
			rustFuture,
			(C.UniffiRustFutureContinuationCallback)(C.blazen_uniffiFutureContinuationCallback),
			C.uint64_t(chanHandle),
		)
		pollResult = <-waiter
	}

	var goValue T
	ffiValue, err := rustCallWithError(errConverter, func(status *C.RustCallStatus) F {
		return completeFunc(rustFuture, status)
	})
	if value := reflect.ValueOf(err); value.IsValid() && !value.IsZero() {
		return goValue, err
	}
	return liftFunc(ffiValue), err
}

//export blazen_uniffiFreeGorutine
func blazen_uniffiFreeGorutine(data C.uint64_t) {
	handle := cgo.Handle(uintptr(data))
	defer handle.Delete()

	guard := handle.Value().(chan struct{})
	guard <- struct{}{}
}

func Version() string {
	return FfiConverterStringINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) RustBufferI {
		return GoRustBuffer{
			inner: C.uniffi_blazen_uniffi_fn_func_version(_uniffiStatus),
		}
	}))
}

// Run a batch of completion requests with bounded concurrency.
//
// - `model`: the model to drive (one provider, one model id; for cross-model
// batches dispatch from foreign code instead).
// - `requests`: the requests to send, in order. Each is converted to the
// upstream wire format before dispatch; conversion errors short-circuit
// the entire batch (the request list is rejected as a whole, since a bad
// schema means the batch was misconfigured).
// - `max_concurrency`: hard cap on in-flight requests. `0` means unlimited
// (all dispatched in parallel).
//
// Returns a [`BatchResult`] with per-request outcomes and aggregated
// usage / cost. Individual request failures appear in
// [`BatchResult::responses`] as [`BatchItem::Failure`] — they do not cause
// this function itself to return `Err`.
//
// # Errors
//
// Returns [`BlazenError::Validation`] if any input request fails to convert
// to the upstream wire format (typically a malformed `parameters_json` or
// `response_format_json` payload).
func CompleteBatch(model *CompletionModel, requests []CompletionRequest, maxConcurrency uint32) (BatchResult, error) {
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) BatchResult {
			return FfiConverterBatchResultINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_func_complete_batch(FfiConverterCompletionModelINSTANCE.Lower(model), FfiConverterSequenceCompletionRequestINSTANCE.Lower(requests), FfiConverterUint32INSTANCE.Lower(maxConcurrency)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_rust_buffer(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_rust_buffer(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// Synchronous variant of [`complete_batch`] — blocks the current thread on
// the shared Tokio runtime.
//
// # Errors
//
// Same as [`complete_batch`].
func CompleteBatchBlocking(model *CompletionModel, requests []CompletionRequest, maxConcurrency uint32) (BatchResult, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) RustBufferI {
		return GoRustBuffer{
			inner: C.uniffi_blazen_uniffi_fn_func_complete_batch_blocking(FfiConverterCompletionModelINSTANCE.Lower(model), FfiConverterSequenceCompletionRequestINSTANCE.Lower(requests), FfiConverterUint32INSTANCE.Lower(maxConcurrency), _uniffiStatus),
		}
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue BatchResult
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterBatchResultINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build a local diffusion-rs image-generation model.
//
// `model_id` is the HuggingFace repo id of the Stable Diffusion variant
// (e.g. `"stabilityai/stable-diffusion-2-1"`). `device` follows the same
// device-string format as the local-LLM factories. `width` / `height` /
// `num_inference_steps` / `guidance_scale` set provider defaults applied
// to every generate call. Calls surface the upstream "engine not yet
// wired" message until the Phase 5.3 work lands; construction succeeds so
// foreign callers can plumb their options today.
func NewDiffusionModel(modelId *string, device *string, width *uint32, height *uint32, numInferenceSteps *uint32, guidanceScale *float32) (*ImageGenModel, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_diffusion_model(FfiConverterOptionalStringINSTANCE.Lower(modelId), FfiConverterOptionalStringINSTANCE.Lower(device), FfiConverterOptionalUint32INSTANCE.Lower(width), FfiConverterOptionalUint32INSTANCE.Lower(height), FfiConverterOptionalUint32INSTANCE.Lower(numInferenceSteps), FfiConverterOptionalFloat32INSTANCE.Lower(guidanceScale), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *ImageGenModel
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterImageGenModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build a fal.ai-backed [`ImageGenModel`].
//
// `api_key` may be empty when the provider resolves it from `FAL_KEY`.
// `model` overrides the default fal image-gen endpoint (e.g.
// `"fal-ai/flux/dev"`); when `None`, fal routes to its current default
// image model. The per-call `model` argument on
// [`ImageGenModel::generate`] takes precedence over this default when
// both are set.
func NewFalImageGenModel(apiKey string, model *string) (*ImageGenModel, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_fal_image_gen_model(FfiConverterStringINSTANCE.Lower(apiKey), FfiConverterOptionalStringINSTANCE.Lower(model), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *ImageGenModel
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterImageGenModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build a fal.ai-backed [`SttModel`].
//
// `api_key` may be empty when the provider resolves it from `FAL_KEY`.
// `model` overrides the default fal transcription endpoint (e.g.
// `"fal-ai/whisper"`); when `None`, fal routes to its current default
// Whisper endpoint.
func NewFalSttModel(apiKey string, model *string) (*SttModel, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_fal_stt_model(FfiConverterStringINSTANCE.Lower(apiKey), FfiConverterOptionalStringINSTANCE.Lower(model), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *SttModel
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterSttModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build a fal.ai-backed [`TtsModel`].
//
// `api_key` may be empty when the provider resolves it from `FAL_KEY`.
// `model` overrides the default fal TTS endpoint (e.g.
// `"fal-ai/dia-tts"`); when `None`, the per-call `voice` / `language`
// arguments decide which endpoint fal routes to.
func NewFalTtsModel(apiKey string, model *string) (*TtsModel, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_fal_tts_model(FfiConverterStringINSTANCE.Lower(apiKey), FfiConverterOptionalStringINSTANCE.Lower(model), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *TtsModel
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterTtsModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build a local Piper text-to-speech model.
//
// `model_id` selects a Piper voice model (e.g. `"en_US-amy-medium"`).
// `speaker_id` selects a speaker for multi-speaker voice models;
// `sample_rate` overrides the model's native sample rate. Returns a
// [`TtsModel`] handle whose [`synthesize`](TtsModel::synthesize) call
// surfaces the upstream "engine not available" error until the Piper
// Phase 9 wiring lands — but construction succeeds so foreign callers
// can wire option plumbing today.
func NewPiperTtsModel(modelId *string, speakerId *uint32, sampleRate *uint32) (*TtsModel, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_piper_tts_model(FfiConverterOptionalStringINSTANCE.Lower(modelId), FfiConverterOptionalUint32INSTANCE.Lower(speakerId), FfiConverterOptionalUint32INSTANCE.Lower(sampleRate), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *TtsModel
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterTtsModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build a local whisper.cpp speech-to-text model.
//
// `model` selects a Whisper variant by name (case-insensitive:
// `"tiny"`, `"base"`, `"small"`, `"medium"`, `"large-v3"`); unrecognised
// values default to `Small`. `device` accepts the same format strings as
// `blazen_llm::Device::parse` (`"cpu"`, `"cuda"`, `"cuda:N"`, `"metal"`).
// `language` is an optional default ISO-639-1 hint (overridable per
// [`SttModel::transcribe`] call).
func NewWhisperSttModel(model *string, device *string, language *string) (*SttModel, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_whisper_stt_model(FfiConverterOptionalStringINSTANCE.Lower(model), FfiConverterOptionalStringINSTANCE.Lower(device), FfiConverterOptionalStringINSTANCE.Lower(language), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *SttModel
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterSttModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build an embedded redb-backed checkpoint store rooted at `path`.
//
// The database file is created if it does not exist. Re-opening an
// existing file is safe and preserves prior checkpoints. The returned
// handle is cheap to clone and safe to share across threads / tasks.
func NewRedbCheckpointStore(path string) (*CheckpointStore, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_redb_checkpoint_store(FfiConverterStringINSTANCE.Lower(path), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *CheckpointStore
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterCheckpointStoreINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build a Redis/ValKey-backed checkpoint store connected to `url`.
//
// `url` is in the form `redis://host:port/db` (or `rediss://` for TLS).
// When `ttl_seconds` is provided every saved checkpoint will auto-expire
// after that many seconds — useful for transient workflows where old
// checkpoints should not accumulate indefinitely.
//
// The initial connection is established eagerly on the shared Tokio
// runtime; subsequent reconnections are handled automatically by the
// underlying connection manager.
//
// # Naming deviation from spec
//
// The task spec named the second argument `namespace`, but
// [`blazen_persist::valkey::ValkeyCheckpointStore`] has no namespace
// concept — instead it supports an optional per-key TTL via
// [`with_ttl`](blazen_persist::valkey::ValkeyCheckpointStore::with_ttl).
// This factory exposes that real option.
func NewValkeyCheckpointStore(url string, ttlSeconds *uint64) (*CheckpointStore, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_valkey_checkpoint_store(FfiConverterStringINSTANCE.Lower(url), FfiConverterOptionalUint64INSTANCE.Lower(ttlSeconds), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *CheckpointStore
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterCheckpointStoreINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build a [`CustomProviderHandle`] from a foreign-implemented
// [`CustomProvider`].
//
// This is the factory foreign users invoke after implementing the
// `CustomProvider` protocol/interface/trait on their own type:
//
// ```kotlin
// class MyProvider : CustomProvider { /* ... 16 methods ... */ }
// val handle = customProviderFromForeign(MyProvider())
// val resp = handle.complete(request)
// ```
//
// The handle holds an internal adapter that converts UniFFI records to
// upstream `blazen_llm::compute` types on each call.
func CustomProviderFromForeign(provider CustomProvider) *CustomProviderHandle {
	return FfiConverterCustomProviderHandleINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_custom_provider_from_foreign(FfiConverterCustomProviderINSTANCE.Lower(provider), _uniffiStatus)
	}))
}

// Convenience: build a [`CustomProviderHandle`] for an LM Studio server.
//
// Equivalent to [`openai_compat`] with `base_url = http://{host}:{port}/v1`
// and no API key. Defaults: `host = "localhost"`, `port = 1234`.
func LmStudio(model string, host *string, port *uint16) *CustomProviderHandle {
	return FfiConverterCustomProviderHandleINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_lm_studio(FfiConverterStringINSTANCE.Lower(model), FfiConverterOptionalStringINSTANCE.Lower(host), FfiConverterOptionalUint16INSTANCE.Lower(port), _uniffiStatus)
	}))
}

// Build a fully-specified [`OpenAiCompatConfig`] from positional arguments.
//
// Convenience for foreign callers that don't want to construct the
// [`OpenAiCompatConfig`] record by hand. Mirrors the
// [`openai_compat`] factory's shape.
func NewOpenaiCompatConfig(providerName string, baseUrl string, apiKey string, defaultModel string, authMethod AuthMethod, supportsModelListing bool) OpenAiCompatConfig {
	return FfiConverterOpenAiCompatConfigINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) RustBufferI {
		return GoRustBuffer{
			inner: C.uniffi_blazen_uniffi_fn_func_new_openai_compat_config(FfiConverterStringINSTANCE.Lower(providerName), FfiConverterStringINSTANCE.Lower(baseUrl), FfiConverterStringINSTANCE.Lower(apiKey), FfiConverterStringINSTANCE.Lower(defaultModel), FfiConverterAuthMethodINSTANCE.Lower(authMethod), FfiConverterBoolINSTANCE.Lower(supportsModelListing), _uniffiStatus),
		}
	}))
}

// Convenience: build a [`CustomProviderHandle`] for a local Ollama server.
//
// Equivalent to [`openai_compat`] with `base_url = http://{host}:{port}/v1`
// and no API key. Defaults: `host = "localhost"`, `port = 11434`.
func Ollama(model string, host *string, port *uint16) *CustomProviderHandle {
	return FfiConverterCustomProviderHandleINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_ollama(FfiConverterStringINSTANCE.Lower(model), FfiConverterOptionalStringINSTANCE.Lower(host), FfiConverterOptionalUint16INSTANCE.Lower(port), _uniffiStatus)
	}))
}

// Build a [`CustomProviderHandle`] for an arbitrary OpenAI-compatible
// backend.
//
// Use for vLLM, llama.cpp's server, TGI, hosted OpenAI-compat services —
// anything that speaks the official OpenAI chat-completions wire format. The
// supplied [`OpenAiCompatConfig`] selects base URL, model, auth method,
// headers, and query parameters.
func OpenaiCompat(providerId string, config OpenAiCompatConfig) *CustomProviderHandle {
	return FfiConverterCustomProviderHandleINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_openai_compat(FfiConverterStringINSTANCE.Lower(providerId), FfiConverterOpenAiCompatConfigINSTANCE.Lower(config), _uniffiStatus)
	}))
}

// Build an Anthropic Messages-API chat-completion model.
func NewAnthropicCompletionModel(apiKey string, model *string, baseUrl *string) (*CompletionModel, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_anthropic_completion_model(FfiConverterStringINSTANCE.Lower(apiKey), FfiConverterOptionalStringINSTANCE.Lower(model), FfiConverterOptionalStringINSTANCE.Lower(baseUrl), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *CompletionModel
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterCompletionModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build an Azure `OpenAI` chat-completion model.
//
// Azure derives its endpoint from `resource_name` + `deployment_name` and
// its model id from `deployment_name`, so `base_url` is intentionally not
// exposed here. `api_version` defaults to the provider's pinned API
// version when `None`.
func NewAzureCompletionModel(apiKey string, resourceName string, deploymentName string, apiVersion *string) (*CompletionModel, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_azure_completion_model(FfiConverterStringINSTANCE.Lower(apiKey), FfiConverterStringINSTANCE.Lower(resourceName), FfiConverterStringINSTANCE.Lower(deploymentName), FfiConverterOptionalStringINSTANCE.Lower(apiVersion), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *CompletionModel
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterCompletionModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build an AWS Bedrock chat-completion model.
//
// `region` selects the AWS region (e.g. `"us-east-1"`); `api_key` is the
// Bedrock API key (which can be obtained via `aws bedrock` IAM keys or
// passed as an empty string to resolve from `AWS_BEARER_TOKEN_BEDROCK`).
func NewBedrockCompletionModel(apiKey string, region string, model *string, baseUrl *string) (*CompletionModel, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_bedrock_completion_model(FfiConverterStringINSTANCE.Lower(apiKey), FfiConverterStringINSTANCE.Lower(region), FfiConverterOptionalStringINSTANCE.Lower(model), FfiConverterOptionalStringINSTANCE.Lower(baseUrl), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *CompletionModel
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterCompletionModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build a local candle chat-completion model.
//
// Wraps [`CandleLlmProvider`](blazen_llm::CandleLlmProvider) through the
// [`CandleLlmCompletionModel`](blazen_llm::CandleLlmCompletionModel) trait
// bridge so it satisfies the same `CompletionModel` trait as remote
// providers.
func NewCandleCompletionModel(modelId string, device *string, quantization *string, revision *string, contextLength *uint32) (*CompletionModel, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_candle_completion_model(FfiConverterStringINSTANCE.Lower(modelId), FfiConverterOptionalStringINSTANCE.Lower(device), FfiConverterOptionalStringINSTANCE.Lower(quantization), FfiConverterOptionalStringINSTANCE.Lower(revision), FfiConverterOptionalUint32INSTANCE.Lower(contextLength), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *CompletionModel
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterCompletionModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build a local candle text-embedding model.
//
// Loads weights from `HuggingFace` and runs inference on-device. Defaults
// to `"sentence-transformers/all-MiniLM-L6-v2"` when `model_id` is `None`.
func NewCandleEmbeddingModel(modelId *string, device *string, revision *string) (*EmbeddingModel, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_candle_embedding_model(FfiConverterOptionalStringINSTANCE.Lower(modelId), FfiConverterOptionalStringINSTANCE.Lower(device), FfiConverterOptionalStringINSTANCE.Lower(revision), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *EmbeddingModel
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterEmbeddingModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build a Cohere chat-completion model.
func NewCohereCompletionModel(apiKey string, model *string, baseUrl *string) (*CompletionModel, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_cohere_completion_model(FfiConverterStringINSTANCE.Lower(apiKey), FfiConverterOptionalStringINSTANCE.Lower(model), FfiConverterOptionalStringINSTANCE.Lower(baseUrl), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *CompletionModel
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterCompletionModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Construct a [`CompletionModel`] that speaks the `OpenAI` chat-completions
// protocol against an arbitrary base URL.
//
// This is the same wire format as
// [`new_openai_compat_completion_model`], but wrapped in a
// [`blazen_llm::CustomProviderHandle`] for consistent ergonomics with the
// `new_ollama_completion_model` / `new_lm_studio_completion_model`
// factories. `api_key` is optional: passing `None` (or an empty `Some`)
// omits the `Authorization` header entirely.
func NewCustomCompletionModelWithOpenaiProtocol(providerId string, baseUrl string, model string, apiKey *string) (*CompletionModel, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_custom_completion_model_with_openai_protocol(FfiConverterStringINSTANCE.Lower(providerId), FfiConverterStringINSTANCE.Lower(baseUrl), FfiConverterStringINSTANCE.Lower(model), FfiConverterOptionalStringINSTANCE.Lower(apiKey), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *CompletionModel
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterCompletionModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build a `DeepSeek` chat-completion model.
func NewDeepseekCompletionModel(apiKey string, model *string, baseUrl *string) (*CompletionModel, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_deepseek_completion_model(FfiConverterStringINSTANCE.Lower(apiKey), FfiConverterOptionalStringINSTANCE.Lower(model), FfiConverterOptionalStringINSTANCE.Lower(baseUrl), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *CompletionModel
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterCompletionModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build a fal.ai chat-completion model.
//
// `endpoint` selects the fal endpoint family — one of
// `"openai_chat"` (default), `"openai_responses"`, `"openai_embeddings"`,
// `"openrouter"`, `"any_llm"`. Unrecognised values fall back to
// `OpenAiChat`. `enterprise` promotes the endpoint to its enterprise /
// SOC2-eligible variant; `auto_route_modality` toggles automatic routing
// to a vision/audio/video endpoint when the request carries media.
func NewFalCompletionModel(apiKey string, model *string, baseUrl *string, endpoint *string, enterprise bool, autoRouteModality bool) (*CompletionModel, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_fal_completion_model(FfiConverterStringINSTANCE.Lower(apiKey), FfiConverterOptionalStringINSTANCE.Lower(model), FfiConverterOptionalStringINSTANCE.Lower(baseUrl), FfiConverterOptionalStringINSTANCE.Lower(endpoint), FfiConverterBoolINSTANCE.Lower(enterprise), FfiConverterBoolINSTANCE.Lower(autoRouteModality), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *CompletionModel
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterCompletionModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build a fal.ai embedding model.
//
// Routes through fal's OpenAI-compatible embeddings endpoint.
// `model` defaults to `"openai/text-embedding-3-small"` (1536 dims);
// `dimensions` overrides the produced vector size (matching the upstream
// model's supported dimensionality).
func NewFalEmbeddingModel(apiKey string, model *string, dimensions *uint32) (*EmbeddingModel, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_fal_embedding_model(FfiConverterStringINSTANCE.Lower(apiKey), FfiConverterOptionalStringINSTANCE.Lower(model), FfiConverterOptionalUint32INSTANCE.Lower(dimensions), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *EmbeddingModel
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterEmbeddingModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build a local fastembed (ONNX Runtime) embedding model.
//
// `model_name` selects a variant from fastembed's catalog (case-insensitive
// debug spelling: `"BGESmallENV15"`, `"AllMiniLML6V2"`, ...). When `None`,
// defaults to `BGESmallENV15`.
func NewFastembedEmbeddingModel(modelName *string, maxBatchSize *uint32, showDownloadProgress *bool) (*EmbeddingModel, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_fastembed_embedding_model(FfiConverterOptionalStringINSTANCE.Lower(modelName), FfiConverterOptionalUint32INSTANCE.Lower(maxBatchSize), FfiConverterOptionalBoolINSTANCE.Lower(showDownloadProgress), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *EmbeddingModel
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterEmbeddingModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build a Fireworks AI chat-completion model.
func NewFireworksCompletionModel(apiKey string, model *string, baseUrl *string) (*CompletionModel, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_fireworks_completion_model(FfiConverterStringINSTANCE.Lower(apiKey), FfiConverterOptionalStringINSTANCE.Lower(model), FfiConverterOptionalStringINSTANCE.Lower(baseUrl), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *CompletionModel
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterCompletionModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build a Google Gemini chat-completion model.
func NewGeminiCompletionModel(apiKey string, model *string, baseUrl *string) (*CompletionModel, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_gemini_completion_model(FfiConverterStringINSTANCE.Lower(apiKey), FfiConverterOptionalStringINSTANCE.Lower(model), FfiConverterOptionalStringINSTANCE.Lower(baseUrl), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *CompletionModel
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterCompletionModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build a Groq chat-completion model.
func NewGroqCompletionModel(apiKey string, model *string, baseUrl *string) (*CompletionModel, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_groq_completion_model(FfiConverterStringINSTANCE.Lower(apiKey), FfiConverterOptionalStringINSTANCE.Lower(model), FfiConverterOptionalStringINSTANCE.Lower(baseUrl), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *CompletionModel
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterCompletionModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build a local llama.cpp chat-completion model.
//
// `model_path` is either a local GGUF file path or a `HuggingFace` repo
// id; `n_gpu_layers` offloads the given number of layers to the GPU when
// the device supports it.
func NewLlamacppCompletionModel(modelPath string, device *string, quantization *string, contextLength *uint32, nGpuLayers *uint32) (*CompletionModel, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_llamacpp_completion_model(FfiConverterStringINSTANCE.Lower(modelPath), FfiConverterOptionalStringINSTANCE.Lower(device), FfiConverterOptionalStringINSTANCE.Lower(quantization), FfiConverterOptionalUint32INSTANCE.Lower(contextLength), FfiConverterOptionalUint32INSTANCE.Lower(nGpuLayers), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *CompletionModel
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterCompletionModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Construct a [`CompletionModel`] for an LM Studio server.
//
// Convenience wrapper around [`blazen_llm::lm_studio`] — targets LM Studio's
// local `OpenAI`-compatible endpoint on `http://{host}:{port}/v1`.
func NewLmStudioCompletionModel(host string, port uint16, model string) (*CompletionModel, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_lm_studio_completion_model(FfiConverterStringINSTANCE.Lower(host), FfiConverterUint16INSTANCE.Lower(port), FfiConverterStringINSTANCE.Lower(model), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *CompletionModel
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterCompletionModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build a Mistral chat-completion model.
func NewMistralCompletionModel(apiKey string, model *string, baseUrl *string) (*CompletionModel, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_mistral_completion_model(FfiConverterStringINSTANCE.Lower(apiKey), FfiConverterOptionalStringINSTANCE.Lower(model), FfiConverterOptionalStringINSTANCE.Lower(baseUrl), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *CompletionModel
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterCompletionModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build a local mistral.rs chat-completion model.
//
// `model_id` is the `HuggingFace` repo id (e.g.
// `"mistralai/Mistral-7B-Instruct-v0.3"`) or a local GGUF path. The
// optional `device`/`quantization` strings follow Blazen's parser format
// (`"cpu"`, `"cuda:0"`, `"metal"`, `"q4_k_m"`, ...). Set `vision = true`
// for multimodal models like LLaVA / Qwen2-VL.
func NewMistralrsCompletionModel(modelId string, device *string, quantization *string, contextLength *uint32, vision bool) (*CompletionModel, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_mistralrs_completion_model(FfiConverterStringINSTANCE.Lower(modelId), FfiConverterOptionalStringINSTANCE.Lower(device), FfiConverterOptionalStringINSTANCE.Lower(quantization), FfiConverterOptionalUint32INSTANCE.Lower(contextLength), FfiConverterBoolINSTANCE.Lower(vision), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *CompletionModel
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterCompletionModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Construct a [`CompletionModel`] for an Ollama server.
//
// Convenience for [`new_custom_completion_model_with_openai_protocol`] with
// `base_url = format!("http://{host}:{port}/v1")` and no API key. Delegates
// to [`blazen_llm::ollama`], which knows how to speak Ollama's flavour of
// the `OpenAI` chat-completions protocol.
func NewOllamaCompletionModel(host string, port uint16, model string) (*CompletionModel, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_ollama_completion_model(FfiConverterStringINSTANCE.Lower(host), FfiConverterUint16INSTANCE.Lower(port), FfiConverterStringINSTANCE.Lower(model), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *CompletionModel
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterCompletionModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build a generic OpenAI-compatible chat-completion model.
//
// Targets any service that speaks the official OpenAI Chat Completions
// wire format (vLLM, llama-server, LM Studio, local proxies, ...). Uses
// `Authorization: Bearer <api_key>` auth.
func NewOpenaiCompatCompletionModel(providerName string, baseUrl string, apiKey string, model string) (*CompletionModel, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_openai_compat_completion_model(FfiConverterStringINSTANCE.Lower(providerName), FfiConverterStringINSTANCE.Lower(baseUrl), FfiConverterStringINSTANCE.Lower(apiKey), FfiConverterStringINSTANCE.Lower(model), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *CompletionModel
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterCompletionModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build an `OpenAI` chat-completion model.
//
// `base_url` defaults to `https://api.openai.com/v1`; override it to target
// any OpenAI-compatible proxy that uses the official-OpenAI request shape.
func NewOpenaiCompletionModel(apiKey string, model *string, baseUrl *string) (*CompletionModel, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_openai_completion_model(FfiConverterStringINSTANCE.Lower(apiKey), FfiConverterOptionalStringINSTANCE.Lower(model), FfiConverterOptionalStringINSTANCE.Lower(baseUrl), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *CompletionModel
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterCompletionModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build an `OpenAI` embedding model.
//
// Defaults to `text-embedding-3-small` (1536 dimensions) when `model` is
// `None`. Passing a custom `model` keeps the model's default
// dimensionality; callers needing a non-default dimensionality should
// drop down to the underlying Rust API.
func NewOpenaiEmbeddingModel(apiKey string, model *string, baseUrl *string) (*EmbeddingModel, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_openai_embedding_model(FfiConverterStringINSTANCE.Lower(apiKey), FfiConverterOptionalStringINSTANCE.Lower(model), FfiConverterOptionalStringINSTANCE.Lower(baseUrl), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *EmbeddingModel
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterEmbeddingModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build an `OpenRouter` chat-completion model.
func NewOpenrouterCompletionModel(apiKey string, model *string, baseUrl *string) (*CompletionModel, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_openrouter_completion_model(FfiConverterStringINSTANCE.Lower(apiKey), FfiConverterOptionalStringINSTANCE.Lower(model), FfiConverterOptionalStringINSTANCE.Lower(baseUrl), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *CompletionModel
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterCompletionModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build a Perplexity chat-completion model.
func NewPerplexityCompletionModel(apiKey string, model *string, baseUrl *string) (*CompletionModel, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_perplexity_completion_model(FfiConverterStringINSTANCE.Lower(apiKey), FfiConverterOptionalStringINSTANCE.Lower(model), FfiConverterOptionalStringINSTANCE.Lower(baseUrl), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *CompletionModel
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterCompletionModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build a Together AI chat-completion model.
func NewTogetherCompletionModel(apiKey string, model *string, baseUrl *string) (*CompletionModel, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_together_completion_model(FfiConverterStringINSTANCE.Lower(apiKey), FfiConverterOptionalStringINSTANCE.Lower(model), FfiConverterOptionalStringINSTANCE.Lower(baseUrl), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *CompletionModel
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterCompletionModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build a local tract (pure-Rust ONNX) embedding model.
//
// Drop-in replacement for [`new_fastembed_embedding_model`] for targets
// where the prebuilt ONNX Runtime binaries can't link (musl-libc, some
// sandboxed environments). Loads the same fastembed model catalog via
// `tract_onnx`.
func NewTractEmbeddingModel(modelName *string, maxBatchSize *uint32, showDownloadProgress *bool) (*EmbeddingModel, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_tract_embedding_model(FfiConverterOptionalStringINSTANCE.Lower(modelName), FfiConverterOptionalUint32INSTANCE.Lower(maxBatchSize), FfiConverterOptionalBoolINSTANCE.Lower(showDownloadProgress), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *EmbeddingModel
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterEmbeddingModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build an xAI (Grok) chat-completion model.
func NewXaiCompletionModel(apiKey string, model *string, baseUrl *string) (*CompletionModel, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_xai_completion_model(FfiConverterStringINSTANCE.Lower(apiKey), FfiConverterOptionalStringINSTANCE.Lower(model), FfiConverterOptionalStringINSTANCE.Lower(baseUrl), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *CompletionModel
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterCompletionModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Eagerly initialise the Tokio runtime and tracing subscriber.
//
// Safe to call multiple times — both initialisations are idempotent.
// Foreign callers typically invoke this once at app startup
// (`blazen.Init()` in Go, `Blazen.initialize()` in Swift, etc.) so the
// first real async call doesn't pay runtime-build latency.
func Init() {
	rustCall(func(_uniffiStatus *C.RustCallStatus) bool {
		C.uniffi_blazen_uniffi_fn_func_init(_uniffiStatus)
		return false
	})
}

// Drive a streaming chat completion, dispatching each chunk to the sink.
//
// On success, calls `sink.on_done(finish_reason, usage)` exactly once and
// returns `Ok(())`. On a provider-side failure (or sink-side
// `on_chunk`/`on_done` failure), calls `sink.on_error(...)` exactly once
// and returns `Ok(())` — the error is *delivered* via the sink, not
// propagated to this function's caller. This keeps the foreign-language
// surface symmetric: the sink owns both happy-path and error-path
// observation.
//
// The only way this function itself returns `Err` is when the initial
// request conversion fails (malformed JSON in tool definitions, etc.) or
// when the upstream `stream()` call fails to *start* the stream. Sink
// callback failures are surfaced via `on_error`.
func CompleteStreaming(model *CompletionModel, request CompletionRequest, sink CompletionStreamSink) error {
	_, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) struct{} {
			C.ffi_blazen_uniffi_rust_future_complete_void(handle, status)
			return struct{}{}
		},
		// liftFn
		func(_ struct{}) struct{} { return struct{}{} },
		C.uniffi_blazen_uniffi_fn_func_complete_streaming(FfiConverterCompletionModelINSTANCE.Lower(model), FfiConverterCompletionRequestINSTANCE.Lower(request), FfiConverterCompletionStreamSinkINSTANCE.Lower(sink)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_void(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_void(handle)
		},
	)

	if err == nil {
		return nil
	}

	return err
}

// Synchronous variant of [`complete_streaming`] — blocks the current
// thread on the shared Tokio runtime.
//
// Handy for Ruby scripts and quick Go main fns where async machinery is
// overkill. The sink's `async` methods still run on the shared runtime
// (they're just driven synchronously from the caller's thread).
func CompleteStreamingBlocking(model *CompletionModel, request CompletionRequest, sink CompletionStreamSink) error {
	_, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) bool {
		C.uniffi_blazen_uniffi_fn_func_complete_streaming_blocking(FfiConverterCompletionModelINSTANCE.Lower(model), FfiConverterCompletionRequestINSTANCE.Lower(request), FfiConverterCompletionStreamSinkINSTANCE.Lower(sink), _uniffiStatus)
		return false
	})
	return _uniffiErr.AsError()
}

// Initialize the Langfuse LLM-observability exporter and install it as the
// global `tracing` subscriber layer.
//
// Spawns a background tokio task that periodically flushes buffered LLM
// call traces, token usage, and latency data to the Langfuse ingestion API.
// Call once at process startup, before any traced work.
//
// Arguments:
// - `public_key`: Langfuse public API key (HTTP Basic-auth username).
// - `secret_key`: Langfuse secret API key (HTTP Basic-auth password).
// - `host`: optional Langfuse host URL; defaults to
// `https://cloud.langfuse.com` when `None`.
//
// Batch size and flush interval use upstream defaults (100 events / 5000 ms).
// If a finer-grained config knob is needed, expose it here later — upstream's
// `LangfuseConfig` supports both via `with_batch_size` / `with_flush_interval_ms`.
//
// If a global `tracing` subscriber is already installed, the underlying
// `LangfuseLayer` is still constructed (so its background dispatcher runs)
// and this returns `Ok(())` without overwriting the existing subscriber.
//
// # Errors
//
// Returns [`BlazenError::Internal`] if the underlying HTTP client or
// dispatcher cannot be built.
func InitLangfuse(publicKey string, secretKey string, host *string) error {
	_, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) bool {
		C.uniffi_blazen_uniffi_fn_func_init_langfuse(FfiConverterStringINSTANCE.Lower(publicKey), FfiConverterStringINSTANCE.Lower(secretKey), FfiConverterOptionalStringINSTANCE.Lower(host), _uniffiStatus)
		return false
	})
	return _uniffiErr.AsError()
}

// Initialize the OpenTelemetry OTLP (gRPC/tonic) trace exporter and install
// it as the global `tracing` subscriber stack.
//
// Arguments:
// - `endpoint`: OTLP gRPC endpoint URL (e.g. `"http://localhost:4317"`).
// - `service_name`: service name reported to the backend; defaults to
// `"blazen"` when `None`.
//
// Upstream's [`blazen_telemetry::OtlpConfig`] does not currently accept
// per-request headers — if your backend needs an `Authorization` header
// (Honeycomb, Datadog, Grafana Cloud, etc.), set it via the
// `OTEL_EXPORTER_OTLP_HEADERS` environment variable, which the
// `opentelemetry-otlp` crate reads at exporter-build time.
//
// # Errors
//
// Returns [`BlazenError::Internal`] if the OTLP exporter or tracer provider
// cannot be constructed.
func InitOtlp(endpoint string, serviceName *string) error {
	_, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) bool {
		C.uniffi_blazen_uniffi_fn_func_init_otlp(FfiConverterStringINSTANCE.Lower(endpoint), FfiConverterOptionalStringINSTANCE.Lower(serviceName), _uniffiStatus)
		return false
	})
	return _uniffiErr.AsError()
}

// Initialize the Prometheus metrics exporter and start the HTTP listener.
//
// Installs a global `metrics` recorder backed by Prometheus and starts an
// HTTP server serving the `/metrics` endpoint.
//
// `listen_address` accepts a `host:port` string (e.g. `"0.0.0.0:9100"`).
// Upstream [`blazen_telemetry::init_prometheus`] always binds `0.0.0.0` and
// only takes a port, so the host portion of `listen_address` is parsed for
// validation but does **not** override the upstream bind address — the
// listener always accepts traffic on every interface. Pass a plain port
// string like `"9100"` to skip the host portion.
//
// # Errors
//
// Returns [`BlazenError::Validation`] if `listen_address` is not a
// well-formed `host:port` (or bare port) string, or
// [`BlazenError::Internal`] if the HTTP listener cannot be bound or the
// global metrics recorder cannot be installed (e.g. one is already set).
func InitPrometheus(listenAddress string) error {
	_, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) bool {
		C.uniffi_blazen_uniffi_fn_func_init_prometheus(FfiConverterStringINSTANCE.Lower(listenAddress), _uniffiStatus)
		return false
	})
	return _uniffiErr.AsError()
}

// Decode a JSON-serialised upstream [`blazen_telemetry::WorkflowHistory`]
// into a flat `Vec<WorkflowHistoryEntry>`.
//
// The expected input is the exact format produced by
// `serde_json::to_string(&history)` on a
// [`blazen_telemetry::WorkflowHistory`] (i.e. an object with `run_id`,
// `workflow_name`, and `events: [{timestamp, sequence, kind}]`). This is
// the same shape the Python binding's `WorkflowHistory.from_json` accepts,
// so foreign callers can round-trip history JSON across bindings.
//
// Returns an empty vector if the history has no events.
//
// `blazen-telemetry`'s `history` feature is hard-pinned on in this crate's
// `Cargo.toml`, so this function is always available regardless of which
// optional exporter features are enabled.
//
// # Errors
//
// Returns [`BlazenError::Validation`] when `history_json` fails to
// deserialise as a [`blazen_telemetry::WorkflowHistory`].
func ParseWorkflowHistory(historyJson string) ([]WorkflowHistoryEntry, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) RustBufferI {
		return GoRustBuffer{
			inner: C.uniffi_blazen_uniffi_fn_func_parse_workflow_history(FfiConverterStringINSTANCE.Lower(historyJson), _uniffiStatus),
		}
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue []WorkflowHistoryEntry
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterSequenceWorkflowHistoryEntryINSTANCE.Lift(_uniffiRV), nil
	}
}

// Best-effort flush + shutdown of any initialised telemetry exporters.
//
// Upstream [`blazen_telemetry`] does not currently expose explicit
// shutdown hooks: the Langfuse dispatcher flushes on its own interval and
// drops cleanly when the process exits; the OTLP `SdkTracerProvider` is
// owned globally by `opentelemetry::global` and flushed on drop; the
// Prometheus listener runs until the process exits. Calling this function
// is therefore safe but currently has no observable effect — it exists so
// foreign callers can wire a single "shutdown" hook into their app
// lifecycle without conditionally branching on features. When upstream
// grows explicit shutdown APIs, this function will route to them.
//
// Safe to call even if no exporter was initialised.
func ShutdownTelemetry() {
	rustCall(func(_uniffiStatus *C.RustCallStatus) bool {
		C.uniffi_blazen_uniffi_fn_func_shutdown_telemetry(_uniffiStatus)
		return false
	})
}
