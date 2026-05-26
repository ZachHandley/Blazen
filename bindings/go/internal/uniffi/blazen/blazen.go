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
		// Defensive: protocol desync between Rust/Go bindgen would leave
		// bytes here. Panicking surfaces it immediately rather than silently
		// dropping data on the floor; keep this guard even though codegen
		// is expected to keep buffers balanced.
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
	FfiConverterControlPlaneAssignmentHandlerINSTANCE.register()
	FfiConverterControlPlaneRunEventSubscriberINSTANCE.register()
	FfiConverterCustomProviderINSTANCE.register()
	FfiConverterForeignLocalModelINSTANCE.register()
	FfiConverterForeignTrainingProgressINSTANCE.register()
	FfiConverterMusicStreamSinkINSTANCE.register()
	FfiConverterStepHandlerINSTANCE.register()
	FfiConverterToolHandlerINSTANCE.register()
	FfiConverterVcStreamSinkINSTANCE.register()
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
		if checksum != 26653 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_complete_batch: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_complete_batch_blocking()
		})
		if checksum != 12966 {
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
			return C.uniffi_blazen_uniffi_checksum_func_new_local_tts_model()
		})
		if checksum != 31651 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_local_tts_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_piper_tts_model()
		})
		if checksum != 62202 {
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
			return C.uniffi_blazen_uniffi_checksum_func_new_audiogen_model()
		})
		if checksum != 43231 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_audiogen_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_fal_music_model()
		})
		if checksum != 18015 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_fal_music_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_musicgen_model()
		})
		if checksum != 54644 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_musicgen_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_stable_audio_model()
		})
		if checksum != 2556 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_stable_audio_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_stream_generate_music_to_sink()
		})
		if checksum != 62355 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_stream_generate_music_to_sink: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_stream_generate_music_to_sink_blocking()
		})
		if checksum != 23279 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_stream_generate_music_to_sink_blocking: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_stream_generate_sfx_to_sink()
		})
		if checksum != 10355 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_stream_generate_sfx_to_sink: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_stream_generate_sfx_to_sink_blocking()
		})
		if checksum != 49910 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_stream_generate_sfx_to_sink_blocking: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_rvc_model()
		})
		if checksum != 51978 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_rvc_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_stream_convert_pcm_to_sink()
		})
		if checksum != 29415 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_stream_convert_pcm_to_sink: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_stream_convert_pcm_to_sink_blocking()
		})
		if checksum != 36415 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_stream_convert_pcm_to_sink_blocking: UniFFI API checksum mismatch")
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
			return C.uniffi_blazen_uniffi_checksum_func_refresh_pricing()
		})
		if checksum != 62705 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_refresh_pricing: UniFFI API checksum mismatch")
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
			return C.uniffi_blazen_uniffi_checksum_func_new_anthropic_model()
		})
		if checksum != 63174 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_anthropic_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_azure_model()
		})
		if checksum != 41519 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_azure_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_bedrock_model()
		})
		if checksum != 25844 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_bedrock_model: UniFFI API checksum mismatch")
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
			return C.uniffi_blazen_uniffi_checksum_func_new_candle_model()
		})
		if checksum != 59984 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_candle_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_cohere_model()
		})
		if checksum != 47421 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_cohere_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_custom_model_with_openai_protocol()
		})
		if checksum != 35849 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_custom_model_with_openai_protocol: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_deepseek_model()
		})
		if checksum != 60107 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_deepseek_model: UniFFI API checksum mismatch")
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
			return C.uniffi_blazen_uniffi_checksum_func_new_fal_model()
		})
		if checksum != 6726 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_fal_model: UniFFI API checksum mismatch")
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
			return C.uniffi_blazen_uniffi_checksum_func_new_fireworks_model()
		})
		if checksum != 25660 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_fireworks_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_gemini_model()
		})
		if checksum != 5451 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_gemini_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_groq_model()
		})
		if checksum != 63063 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_groq_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_llamacpp_model()
		})
		if checksum != 62567 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_llamacpp_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_lm_studio_model()
		})
		if checksum != 54376 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_lm_studio_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_mistral_model()
		})
		if checksum != 18547 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_mistral_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_mistralrs_model()
		})
		if checksum != 56474 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_mistralrs_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_ollama_model()
		})
		if checksum != 51199 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_ollama_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_openai_compat_model()
		})
		if checksum != 1505 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_openai_compat_model: UniFFI API checksum mismatch")
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
			return C.uniffi_blazen_uniffi_checksum_func_new_openai_model()
		})
		if checksum != 52484 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_openai_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_openrouter_model()
		})
		if checksum != 25099 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_openrouter_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_perplexity_model()
		})
		if checksum != 49956 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_perplexity_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_new_together_model()
		})
		if checksum != 60884 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_together_model: UniFFI API checksum mismatch")
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
			return C.uniffi_blazen_uniffi_checksum_func_new_xai_model()
		})
		if checksum != 50122 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_new_xai_model: UniFFI API checksum mismatch")
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
		if checksum != 22515 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_func_complete_streaming: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_func_complete_streaming_blocking()
		})
		if checksum != 64487 {
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
		if checksum != 37809 {
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
			return C.uniffi_blazen_uniffi_checksum_method_musicmodel_generate_music()
		})
		if checksum != 60700 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_musicmodel_generate_music: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_musicmodel_generate_music_blocking()
		})
		if checksum != 33543 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_musicmodel_generate_music_blocking: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_musicmodel_generate_sfx()
		})
		if checksum != 21245 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_musicmodel_generate_sfx: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_musicmodel_generate_sfx_blocking()
		})
		if checksum != 60492 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_musicmodel_generate_sfx_blocking: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_musicstreamsink_on_chunk()
		})
		if checksum != 7832 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_musicstreamsink_on_chunk: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_musicstreamsink_on_done()
		})
		if checksum != 61428 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_musicstreamsink_on_done: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_musicstreamsink_on_error()
		})
		if checksum != 44317 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_musicstreamsink_on_error: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_vcmodel_convert_voice()
		})
		if checksum != 46177 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_vcmodel_convert_voice: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_vcmodel_convert_voice_blocking()
		})
		if checksum != 59967 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_vcmodel_convert_voice_blocking: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_vcmodel_list_target_voices()
		})
		if checksum != 2307 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_vcmodel_list_target_voices: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_vcmodel_list_target_voices_blocking()
		})
		if checksum != 32701 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_vcmodel_list_target_voices_blocking: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_vcmodel_register_target_voice()
		})
		if checksum != 15373 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_vcmodel_register_target_voice: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_vcmodel_register_target_voice_blocking()
		})
		if checksum != 31343 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_vcmodel_register_target_voice_blocking: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_vcstreamsink_on_chunk()
		})
		if checksum != 1538 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_vcstreamsink_on_chunk: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_vcstreamsink_on_done()
		})
		if checksum != 20371 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_vcstreamsink_on_done: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_vcstreamsink_on_error()
		})
		if checksum != 5727 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_vcstreamsink_on_error: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_controlplaneassignmenthandler_handle()
		})
		if checksum != 640 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_controlplaneassignmenthandler_handle: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_controlplaneassignmenthandler_on_cancel()
		})
		if checksum != 36399 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_controlplaneassignmenthandler_on_cancel: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_controlplaneassignmenthandler_on_drain()
		})
		if checksum != 63250 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_controlplaneassignmenthandler_on_drain: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_controlplaneclient_cancel_workflow()
		})
		if checksum != 39238 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_controlplaneclient_cancel_workflow: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_controlplaneclient_describe_workflow()
		})
		if checksum != 8003 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_controlplaneclient_describe_workflow: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_controlplaneclient_drain_worker()
		})
		if checksum != 17174 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_controlplaneclient_drain_worker: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_controlplaneclient_list_workers()
		})
		if checksum != 315 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_controlplaneclient_list_workers: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_controlplaneclient_submit_workflow()
		})
		if checksum != 23792 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_controlplaneclient_submit_workflow: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_controlplaneclient_subscribe_run_events()
		})
		if checksum != 40711 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_controlplaneclient_subscribe_run_events: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_controlplaneruneventsubscriber_on_event()
		})
		if checksum != 3038 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_controlplaneruneventsubscriber_on_event: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_controlplaneruneventsubscriber_on_close()
		})
		if checksum != 46575 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_controlplaneruneventsubscriber_on_close: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_controlplaneruneventsubscriber_on_error()
		})
		if checksum != 32772 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_controlplaneruneventsubscriber_on_error: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_controlplanesubscription_cancel()
		})
		if checksum != 27973 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_controlplanesubscription_cancel: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_controlplaneworker_run()
		})
		if checksum != 52241 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_controlplaneworker_run: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_controlplaneworker_shutdown()
		})
		if checksum != 13840 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_controlplaneworker_shutdown: UniFFI API checksum mismatch")
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
			return C.uniffi_blazen_uniffi_checksum_method_model_complete()
		})
		if checksum != 48637 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_model_complete: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_model_complete_blocking()
		})
		if checksum != 36439 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_model_complete_blocking: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_model_model_id()
		})
		if checksum != 39325 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_model_model_id: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_foreignlocalmodel_load()
		})
		if checksum != 57606 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_foreignlocalmodel_load: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_foreignlocalmodel_unload()
		})
		if checksum != 30655 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_foreignlocalmodel_unload: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_foreignlocalmodel_is_loaded()
		})
		if checksum != 35239 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_foreignlocalmodel_is_loaded: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_foreignlocalmodel_device()
		})
		if checksum != 36918 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_foreignlocalmodel_device: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_foreignlocalmodel_memory_bytes()
		})
		if checksum != 63180 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_foreignlocalmodel_memory_bytes: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_foreignlocalmodel_load_adapter()
		})
		if checksum != 64869 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_foreignlocalmodel_load_adapter: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_foreignlocalmodel_unload_adapter()
		})
		if checksum != 45180 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_foreignlocalmodel_unload_adapter: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_foreignlocalmodel_list_adapters()
		})
		if checksum != 49821 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_foreignlocalmodel_list_adapters: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_uniffimodelmanager_available_bytes()
		})
		if checksum != 19672 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_uniffimodelmanager_available_bytes: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_uniffimodelmanager_ensure_loaded()
		})
		if checksum != 33632 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_uniffimodelmanager_ensure_loaded: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_uniffimodelmanager_is_loaded()
		})
		if checksum != 48692 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_uniffimodelmanager_is_loaded: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_uniffimodelmanager_list_adapters()
		})
		if checksum != 57369 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_uniffimodelmanager_list_adapters: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_uniffimodelmanager_load()
		})
		if checksum != 56552 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_uniffimodelmanager_load: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_uniffimodelmanager_load_adapter()
		})
		if checksum != 2343 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_uniffimodelmanager_load_adapter: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_uniffimodelmanager_load_blocking()
		})
		if checksum != 10510 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_uniffimodelmanager_load_blocking: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_uniffimodelmanager_load_from_hf()
		})
		if checksum != 38990 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_uniffimodelmanager_load_from_hf: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_uniffimodelmanager_pools()
		})
		if checksum != 34891 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_uniffimodelmanager_pools: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_uniffimodelmanager_register_local()
		})
		if checksum != 23428 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_uniffimodelmanager_register_local: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_uniffimodelmanager_status()
		})
		if checksum != 18811 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_uniffimodelmanager_status: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_uniffimodelmanager_unload()
		})
		if checksum != 3272 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_uniffimodelmanager_unload: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_uniffimodelmanager_unload_adapter()
		})
		if checksum != 43755 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_uniffimodelmanager_unload_adapter: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_uniffimodelmanager_unload_blocking()
		})
		if checksum != 2783 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_uniffimodelmanager_unload_blocking: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_uniffimodelmanager_used_bytes()
		})
		if checksum != 51636 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_uniffimodelmanager_used_bytes: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_uniffimodelmanager_fine_tune()
		})
		if checksum != 37026 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_uniffimodelmanager_fine_tune: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_uniffimodelmanager_train_dpo()
		})
		if checksum != 58114 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_uniffimodelmanager_train_dpo: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_uniffimodelmanager_train_kto()
		})
		if checksum != 61821 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_uniffimodelmanager_train_kto: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_uniffimodelmanager_train_lora()
		})
		if checksum != 22343 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_uniffimodelmanager_train_lora: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_uniffimodelmanager_train_orpo()
		})
		if checksum != 40322 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_uniffimodelmanager_train_orpo: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_uniffimodelmanager_train_simpo()
		})
		if checksum != 57801 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_uniffimodelmanager_train_simpo: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_foreigntrainingprogress_on_event()
		})
		if checksum != 34103 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_foreigntrainingprogress_on_event: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_uniffijsonldataset_is_empty()
		})
		if checksum != 21289 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_uniffijsonldataset_is_empty: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_uniffijsonldataset_len()
		})
		if checksum != 23707 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_uniffijsonldataset_len: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_uniffipreferencejsonldataset_is_empty()
		})
		if checksum != 24611 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_uniffipreferencejsonldataset_is_empty: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_uniffipreferencejsonldataset_len()
		})
		if checksum != 52745 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_uniffipreferencejsonldataset_len: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_uniffiratedjsonldataset_is_empty()
		})
		if checksum != 34789 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_uniffiratedjsonldataset_is_empty: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_uniffiratedjsonldataset_len()
		})
		if checksum != 48605 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_uniffiratedjsonldataset_len: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_modelclient_complete()
		})
		if checksum != 42969 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_modelclient_complete: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_modelclient_embed()
		})
		if checksum != 27994 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_modelclient_embed: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_modelclient_fetch_blob()
		})
		if checksum != 31110 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_modelclient_fetch_blob: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_modelclient_generate_image()
		})
		if checksum != 64354 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_modelclient_generate_image: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_modelclient_generate_music()
		})
		if checksum != 17866 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_modelclient_generate_music: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_modelclient_is_loaded()
		})
		if checksum != 2818 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_modelclient_is_loaded: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_modelclient_list_adapters()
		})
		if checksum != 9453 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_modelclient_list_adapters: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_modelclient_load()
		})
		if checksum != 64540 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_modelclient_load: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_modelclient_load_adapter()
		})
		if checksum != 33876 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_modelclient_load_adapter: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_modelclient_load_from_hf()
		})
		if checksum != 27748 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_modelclient_load_from_hf: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_modelclient_status()
		})
		if checksum != 46116 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_modelclient_status: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_modelclient_stream_complete()
		})
		if checksum != 49585 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_modelclient_stream_complete: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_modelclient_text_to_speech()
		})
		if checksum != 48656 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_modelclient_text_to_speech: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_modelclient_transcribe()
		})
		if checksum != 48760 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_modelclient_transcribe: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_modelclient_unload()
		})
		if checksum != 6162 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_modelclient_unload: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_modelclient_unload_adapter()
		})
		if checksum != 26088 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_modelclient_unload_adapter: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_modelclient_upload_blob()
		})
		if checksum != 1673 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_modelclient_upload_blob: UniFFI API checksum mismatch")
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
			return C.uniffi_blazen_uniffi_checksum_method_baseprovider_as_model()
		})
		if checksum != 58666 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_baseprovider_as_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_baseprovider_defaults()
		})
		if checksum != 21474 {
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
		if checksum != 18282 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_baseprovider_model_id: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_baseprovider_with_defaults()
		})
		if checksum != 47332 {
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
		if checksum != 59926 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_method_customprovider_complete: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_method_customprovider_stream()
		})
		if checksum != 9479 {
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
		if checksum != 43943 {
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
		if checksum != 24656 {
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
		if checksum != 57273 {
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
		if checksum != 59656 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_constructor_agent_new: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_constructor_controlplaneclient_connect()
		})
		if checksum != 13355 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_constructor_controlplaneclient_connect: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_constructor_controlplaneclient_connect_blocking()
		})
		if checksum != 65181 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_constructor_controlplaneclient_connect_blocking: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_constructor_controlplaneworker_new_blocking()
		})
		if checksum != 22162 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_constructor_controlplaneworker_new_blocking: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_constructor_uniffimodelmanager_new()
		})
		if checksum != 31159 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_constructor_uniffimodelmanager_new: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_constructor_uniffimodelmanager_with_budgets_gb()
		})
		if checksum != 1375 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_constructor_uniffimodelmanager_with_budgets_gb: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_constructor_uniffimodelmanager_with_pool_budgets()
		})
		if checksum != 59591 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_constructor_uniffimodelmanager_with_pool_budgets: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_constructor_uniffijsonldataset_from_path()
		})
		if checksum != 8101 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_constructor_uniffijsonldataset_from_path: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_constructor_uniffipreferencejsonldataset_from_path()
		})
		if checksum != 2096 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_constructor_uniffipreferencejsonldataset_from_path: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_constructor_uniffiratedjsonldataset_from_path()
		})
		if checksum != 20887 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_constructor_uniffiratedjsonldataset_from_path: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_constructor_modelclient_connect()
		})
		if checksum != 48616 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_constructor_modelclient_connect: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_constructor_modelclient_connect_with_tls()
		})
		if checksum != 61737 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_constructor_modelclient_connect_with_tls: UniFFI API checksum mismatch")
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
			return C.uniffi_blazen_uniffi_checksum_constructor_baseprovider_from_model()
		})
		if checksum != 28877 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_constructor_baseprovider_from_model: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_blazen_uniffi_checksum_constructor_baseprovider_from_model_with_defaults()
		})
		if checksum != 44881 {
			// If this happens try cleaning and rebuilding your project
			panic("blazen: uniffi_blazen_uniffi_checksum_constructor_baseprovider_from_model_with_defaults: UniFFI API checksum mismatch")
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

type FfiConverterBytes struct{}

var FfiConverterBytesINSTANCE = FfiConverterBytes{}

func (c FfiConverterBytes) Lower(value []byte) C.RustBuffer {
	return LowerIntoRustBuffer[[]byte](c, value)
}

func (c FfiConverterBytes) LowerExternal(value []byte) ExternalCRustBuffer {
	return RustBufferFromC(c.Lower(value))
}

func (c FfiConverterBytes) Write(writer io.Writer, value []byte) {
	if len(value) > math.MaxInt32 {
		panic("[]byte is too large to fit into Int32")
	}

	writeInt32(writer, int32(len(value)))
	write_length, err := writer.Write(value)
	if err != nil {
		panic(err)
	}
	if write_length != len(value) {
		panic(fmt.Errorf("bad write length when writing []byte, expected %d, written %d", len(value), write_length))
	}
}

func (c FfiConverterBytes) Lift(rb RustBufferI) []byte {
	return LiftFromRustBuffer[[]byte](c, rb)
}

func (c FfiConverterBytes) Read(reader io.Reader) []byte {
	length := readInt32(reader)
	buffer := make([]byte, length)
	read_length, err := reader.Read(buffer)
	if err != nil && err != io.EOF {
		panic(err)
	}
	if read_length != int(length) {
		panic(fmt.Errorf("bad read length when reading []byte, expected %d, read %d", length, read_length))
	}
	return buffer
}

type FfiDestroyerBytes struct{}

func (FfiDestroyerBytes) Destroy(_ []byte) {}

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
func NewAgent(model *Model, systemPrompt *string, tools []Tool, toolHandler ToolHandler, maxIterations uint32) *Agent {
	return FfiConverterAgentINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_constructor_agent_new(FfiConverterModelINSTANCE.Lower(model), FfiConverterOptionalStringINSTANCE.Lower(systemPrompt), FfiConverterSequenceToolINSTANCE.Lower(tools), FfiConverterToolHandlerINSTANCE.Lower(toolHandler), FfiConverterUint32INSTANCE.Lower(maxIterations), _uniffiStatus)
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
	// SAFETY (audited 2026-05-13): incrementPointer calls cloneFunction
	// which does Arc::clone on the Rust side, bumping the Rust refcount
	// independently of the Go-side callCounter. The defer below only
	// decrements the (redundant) Go counter; the returned handle survives
	// because the C caller owns its own Arc refcount via Arc::from_raw.
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

// A [`crate::llm::Model`] wrapped with applied
// [`ProviderDefaults`].
//
// Construct via [`BaseProvider::from_model`] (wraps an existing
// model with no defaults) or [`BaseProvider::with_defaults`]
// (wraps with explicit defaults). Mutate via the `with_*` builder methods.
//
// Phase B's `CustomProvider` factories will return `Arc<BaseProvider>`
// directly; for Phase A this class is reachable by lifting any existing
// `Model` factory result.
type BaseProviderInterface interface {
	// Unwrap to a plain [`Model`] handle that applies the
	// configured defaults on every call.
	//
	// Use this when you want to pass the wrapped provider to an API that
	// takes a generic `Model` (the agent runner, workflow
	// steps, etc.).
	AsModel() *Model
	// Inspect the currently-configured defaults (data only — hooks are
	// not surfaced in Phase A).
	Defaults() ProviderDefaults
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
	// The model id of the wrapped inner `Model`.
	ModelId() string
	// Replace the entire [`ProviderDefaults`] on this provider,
	// returning a new `Arc<BaseProvider>` (clone-with-mutation).
	WithDefaults(defaults ProviderDefaults) *BaseProvider
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

// A [`crate::llm::Model`] wrapped with applied
// [`ProviderDefaults`].
//
// Construct via [`BaseProvider::from_model`] (wraps an existing
// model with no defaults) or [`BaseProvider::with_defaults`]
// (wraps with explicit defaults). Mutate via the `with_*` builder methods.
//
// Phase B's `CustomProvider` factories will return `Arc<BaseProvider>`
// directly; for Phase A this class is reachable by lifting any existing
// `Model` factory result.
type BaseProvider struct {
	ffiObject FfiObject
}

// Wrap an existing [`Model`] with empty defaults.
//
// Equivalent to using the wrapped model directly, but lets callers
// attach defaults later via the `with_*` methods.
func BaseProviderFromModel(model *Model) *BaseProvider {
	return FfiConverterBaseProviderINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_constructor_baseprovider_from_model(FfiConverterModelINSTANCE.Lower(model), _uniffiStatus)
	}))
}

// Wrap a [`Model`] with explicit
// [`ProviderDefaults`].
func BaseProviderFromModelWithDefaults(model *Model, defaults ProviderDefaults) *BaseProvider {
	return FfiConverterBaseProviderINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_constructor_baseprovider_from_model_with_defaults(FfiConverterModelINSTANCE.Lower(model), FfiConverterProviderDefaultsINSTANCE.Lower(defaults), _uniffiStatus)
	}))
}

// Unwrap to a plain [`Model`] handle that applies the
// configured defaults on every call.
//
// Use this when you want to pass the wrapped provider to an API that
// takes a generic `Model` (the agent runner, workflow
// steps, etc.).
func (_self *BaseProvider) AsModel() *Model {
	_pointer := _self.ffiObject.incrementPointer("*BaseProvider")
	defer _self.ffiObject.decrementPointer()
	return FfiConverterModelINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_method_baseprovider_as_model(
			_pointer, _uniffiStatus)
	}))
}

// Inspect the currently-configured defaults (data only — hooks are
// not surfaced in Phase A).
func (_self *BaseProvider) Defaults() ProviderDefaults {
	_pointer := _self.ffiObject.incrementPointer("*BaseProvider")
	defer _self.ffiObject.decrementPointer()
	return FfiConverterProviderDefaultsINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) RustBufferI {
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

// The model id of the wrapped inner `Model`.
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

// Replace the entire [`ProviderDefaults`] on this provider,
// returning a new `Arc<BaseProvider>` (clone-with-mutation).
func (_self *BaseProvider) WithDefaults(defaults ProviderDefaults) *BaseProvider {
	_pointer := _self.ffiObject.incrementPointer("*BaseProvider")
	defer _self.ffiObject.decrementPointer()
	return FfiConverterBaseProviderINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_method_baseprovider_with_defaults(
			_pointer, FfiConverterProviderDefaultsINSTANCE.Lower(defaults), _uniffiStatus)
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
	// SAFETY (audited 2026-05-13): incrementPointer calls cloneFunction
	// which does Arc::clone on the Rust side, bumping the Rust refcount
	// independently of the Go-side callCounter. The defer below only
	// decrements the (redundant) Go counter; the returned handle survives
	// because the C caller owns its own Arc refcount via Arc::from_raw.
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
	// SAFETY (audited 2026-05-13): incrementPointer calls cloneFunction
	// which does Arc::clone on the Rust side, bumping the Rust refcount
	// independently of the Go-side callCounter. The defer below only
	// decrements the (redundant) Go counter; the returned handle survives
	// because the C caller owns its own Arc refcount via Arc::from_raw.
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
	// SAFETY (audited 2026-05-13): incrementPointer calls cloneFunction
	// which does Arc::clone on the Rust side, bumping the Rust refcount
	// independently of the Go-side callCounter. The defer below only
	// decrements the (redundant) Go counter; the returned handle survives
	// because the C caller owns its own Arc refcount via Arc::from_raw.
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

// Foreign-implementable handler invoked once per assignment a worker
// receives.
//
// `handle` is synchronous on the foreign side — foreign code that needs
// to drive its own async work must spawn a goroutine / coroutine /
// fiber inside the callback and return when that work resolves. The
// returned `Ok(String)` is interpreted as the assignment output's JSON
// representation; the returned `Err(String)` is surfaced to the control
// plane as an assignment failure.
//
// `on_cancel` and `on_drain` are best-effort notifications. The
// underlying Rust worker has already fired the per-run cancellation
// token / queue gate before invoking these; the foreign handler should
// use them only to release external resources (open file handles,
// network sockets, etc.).
type ControlPlaneAssignmentHandler interface {
	// Handle one assignment. Return `Ok(json)` for success or any
	// [`BlazenError`] for failure — the error's `Display`
	// representation is forwarded to the control plane as the
	// assignment failure message.
	//
	// Use [`BlazenError::Tool`] for handler-side errors, or
	// [`BlazenError::Workflow`] for workflow-level failures.
	Handle(runId string, workflowName string, inputJson string) (string, error)
	// Called when the server cancels an in-flight run. Foreign code
	// should treat this as a notification; the underlying Rust worker
	// has already fired the per-run cancellation token.
	OnCancel(runId string)
	// Called when the server initiates a drain. `immediate = true`
	// means the worker must stop now; `false` means graceful drain.
	OnDrain(immediate bool)
}

// Foreign-implementable handler invoked once per assignment a worker
// receives.
//
// `handle` is synchronous on the foreign side — foreign code that needs
// to drive its own async work must spawn a goroutine / coroutine /
// fiber inside the callback and return when that work resolves. The
// returned `Ok(String)` is interpreted as the assignment output's JSON
// representation; the returned `Err(String)` is surfaced to the control
// plane as an assignment failure.
//
// `on_cancel` and `on_drain` are best-effort notifications. The
// underlying Rust worker has already fired the per-run cancellation
// token / queue gate before invoking these; the foreign handler should
// use them only to release external resources (open file handles,
// network sockets, etc.).
type ControlPlaneAssignmentHandlerImpl struct {
	ffiObject FfiObject
}

// Handle one assignment. Return `Ok(json)` for success or any
// [`BlazenError`] for failure — the error's `Display`
// representation is forwarded to the control plane as the
// assignment failure message.
//
// Use [`BlazenError::Tool`] for handler-side errors, or
// [`BlazenError::Workflow`] for workflow-level failures.
func (_self *ControlPlaneAssignmentHandlerImpl) Handle(runId string, workflowName string, inputJson string) (string, error) {
	_pointer := _self.ffiObject.incrementPointer("ControlPlaneAssignmentHandler")
	defer _self.ffiObject.decrementPointer()
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) RustBufferI {
		return GoRustBuffer{
			inner: C.uniffi_blazen_uniffi_fn_method_controlplaneassignmenthandler_handle(
				_pointer, FfiConverterStringINSTANCE.Lower(runId), FfiConverterStringINSTANCE.Lower(workflowName), FfiConverterStringINSTANCE.Lower(inputJson), _uniffiStatus),
		}
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue string
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterStringINSTANCE.Lift(_uniffiRV), nil
	}
}

// Called when the server cancels an in-flight run. Foreign code
// should treat this as a notification; the underlying Rust worker
// has already fired the per-run cancellation token.
func (_self *ControlPlaneAssignmentHandlerImpl) OnCancel(runId string) {
	_pointer := _self.ffiObject.incrementPointer("ControlPlaneAssignmentHandler")
	defer _self.ffiObject.decrementPointer()
	rustCall(func(_uniffiStatus *C.RustCallStatus) bool {
		C.uniffi_blazen_uniffi_fn_method_controlplaneassignmenthandler_on_cancel(
			_pointer, FfiConverterStringINSTANCE.Lower(runId), _uniffiStatus)
		return false
	})
}

// Called when the server initiates a drain. `immediate = true`
// means the worker must stop now; `false` means graceful drain.
func (_self *ControlPlaneAssignmentHandlerImpl) OnDrain(immediate bool) {
	_pointer := _self.ffiObject.incrementPointer("ControlPlaneAssignmentHandler")
	defer _self.ffiObject.decrementPointer()
	rustCall(func(_uniffiStatus *C.RustCallStatus) bool {
		C.uniffi_blazen_uniffi_fn_method_controlplaneassignmenthandler_on_drain(
			_pointer, FfiConverterBoolINSTANCE.Lower(immediate), _uniffiStatus)
		return false
	})
}
func (object *ControlPlaneAssignmentHandlerImpl) Destroy() {
	runtime.SetFinalizer(object, nil)
	object.ffiObject.destroy()
}

type FfiConverterControlPlaneAssignmentHandler struct {
	handleMap *concurrentHandleMap[ControlPlaneAssignmentHandler]
}

var FfiConverterControlPlaneAssignmentHandlerINSTANCE = FfiConverterControlPlaneAssignmentHandler{
	handleMap: newConcurrentHandleMap[ControlPlaneAssignmentHandler](),
}

func (c FfiConverterControlPlaneAssignmentHandler) Lift(handle C.uint64_t) ControlPlaneAssignmentHandler {
	if uint64(handle)&1 == 0 {
		// Rust-generated handle (even), construct a new object wrapping the handle
		result := &ControlPlaneAssignmentHandlerImpl{
			newFfiObject(
				handle,
				func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
					return C.uniffi_blazen_uniffi_fn_clone_controlplaneassignmenthandler(handle, status)
				},
				func(handle C.uint64_t, status *C.RustCallStatus) {
					C.uniffi_blazen_uniffi_fn_free_controlplaneassignmenthandler(handle, status)
				},
			),
		}
		runtime.SetFinalizer(result, (*ControlPlaneAssignmentHandlerImpl).Destroy)
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

func (c FfiConverterControlPlaneAssignmentHandler) Read(reader io.Reader) ControlPlaneAssignmentHandler {
	return c.Lift(C.uint64_t(readUint64(reader)))
}

func (c FfiConverterControlPlaneAssignmentHandler) Lower(value ControlPlaneAssignmentHandler) C.uint64_t {
	// SAFETY (audited 2026-05-13): incrementPointer calls cloneFunction
	// which does Arc::clone on the Rust side, bumping the Rust refcount
	// independently of the Go-side callCounter. The defer below only
	// decrements the (redundant) Go counter; the returned handle survives
	// because the C caller owns its own Arc refcount via Arc::from_raw.
	if val, ok := value.(*ControlPlaneAssignmentHandlerImpl); ok {
		// Rust-backed object, clone the handle
		handle := val.ffiObject.incrementPointer("ControlPlaneAssignmentHandler")
		defer val.ffiObject.decrementPointer()
		return handle
	} else {
		// Go-backed object, insert into handle map
		return C.uint64_t(c.handleMap.insert(value))
	}
}

func (c FfiConverterControlPlaneAssignmentHandler) Write(writer io.Writer, value ControlPlaneAssignmentHandler) {
	writeUint64(writer, uint64(c.Lower(value)))
}

func LiftFromExternalControlPlaneAssignmentHandler(handle uint64) ControlPlaneAssignmentHandler {
	return FfiConverterControlPlaneAssignmentHandlerINSTANCE.Lift(C.uint64_t(handle))
}

func LowerToExternalControlPlaneAssignmentHandler(value ControlPlaneAssignmentHandler) uint64 {
	return uint64(FfiConverterControlPlaneAssignmentHandlerINSTANCE.Lower(value))
}

type FfiDestroyerControlPlaneAssignmentHandler struct{}

func (_ FfiDestroyerControlPlaneAssignmentHandler) Destroy(value ControlPlaneAssignmentHandler) {
	if val, ok := value.(*ControlPlaneAssignmentHandlerImpl); ok {
		val.Destroy()
	}
}

//export blazen_uniffi_controlplane_cgo_dispatchCallbackInterfaceControlPlaneAssignmentHandlerMethod0
func blazen_uniffi_controlplane_cgo_dispatchCallbackInterfaceControlPlaneAssignmentHandlerMethod0(uniffiHandle C.uint64_t, runId C.RustBuffer, workflowName C.RustBuffer, inputJson C.RustBuffer, uniffiOutReturn *C.RustBuffer, callStatus *C.RustCallStatus) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterControlPlaneAssignmentHandlerINSTANCE.handleMap.tryGet(handle)
	if !ok {
		panic(fmt.Errorf("no callback in handle map: %d", handle))
	}

	res, err :=
		uniffiObj.Handle(
			FfiConverterStringINSTANCE.Lift(GoRustBuffer{
				inner: runId,
			}),
			FfiConverterStringINSTANCE.Lift(GoRustBuffer{
				inner: workflowName,
			}),
			FfiConverterStringINSTANCE.Lift(GoRustBuffer{
				inner: inputJson,
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
}

//export blazen_uniffi_controlplane_cgo_dispatchCallbackInterfaceControlPlaneAssignmentHandlerMethod1
func blazen_uniffi_controlplane_cgo_dispatchCallbackInterfaceControlPlaneAssignmentHandlerMethod1(uniffiHandle C.uint64_t, runId C.RustBuffer, uniffiOutReturn *C.void, callStatus *C.RustCallStatus) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterControlPlaneAssignmentHandlerINSTANCE.handleMap.tryGet(handle)
	if !ok {
		panic(fmt.Errorf("no callback in handle map: %d", handle))
	}

	uniffiObj.OnCancel(
		FfiConverterStringINSTANCE.Lift(GoRustBuffer{
			inner: runId,
		}),
	)

}

//export blazen_uniffi_controlplane_cgo_dispatchCallbackInterfaceControlPlaneAssignmentHandlerMethod2
func blazen_uniffi_controlplane_cgo_dispatchCallbackInterfaceControlPlaneAssignmentHandlerMethod2(uniffiHandle C.uint64_t, immediate C.int8_t, uniffiOutReturn *C.void, callStatus *C.RustCallStatus) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterControlPlaneAssignmentHandlerINSTANCE.handleMap.tryGet(handle)
	if !ok {
		panic(fmt.Errorf("no callback in handle map: %d", handle))
	}

	uniffiObj.OnDrain(
		FfiConverterBoolINSTANCE.Lift(immediate),
	)

}

var UniffiVTableCallbackInterfaceControlPlaneAssignmentHandlerINSTANCE = C.UniffiVTableCallbackInterfaceControlPlaneAssignmentHandler{
	uniffiFree:  (C.UniffiCallbackInterfaceFree)(C.blazen_uniffi_controlplane_cgo_dispatchCallbackInterfaceControlPlaneAssignmentHandlerFree),
	uniffiClone: (C.UniffiCallbackInterfaceClone)(C.blazen_uniffi_controlplane_cgo_dispatchCallbackInterfaceControlPlaneAssignmentHandlerClone),
	handle:      (C.UniffiCallbackInterfaceControlPlaneAssignmentHandlerMethod0)(C.blazen_uniffi_controlplane_cgo_dispatchCallbackInterfaceControlPlaneAssignmentHandlerMethod0),
	onCancel:    (C.UniffiCallbackInterfaceControlPlaneAssignmentHandlerMethod1)(C.blazen_uniffi_controlplane_cgo_dispatchCallbackInterfaceControlPlaneAssignmentHandlerMethod1),
	onDrain:     (C.UniffiCallbackInterfaceControlPlaneAssignmentHandlerMethod2)(C.blazen_uniffi_controlplane_cgo_dispatchCallbackInterfaceControlPlaneAssignmentHandlerMethod2),
}

//export blazen_uniffi_controlplane_cgo_dispatchCallbackInterfaceControlPlaneAssignmentHandlerFree
func blazen_uniffi_controlplane_cgo_dispatchCallbackInterfaceControlPlaneAssignmentHandlerFree(handle C.uint64_t) {
	FfiConverterControlPlaneAssignmentHandlerINSTANCE.handleMap.remove(uint64(handle))
}

//export blazen_uniffi_controlplane_cgo_dispatchCallbackInterfaceControlPlaneAssignmentHandlerClone
func blazen_uniffi_controlplane_cgo_dispatchCallbackInterfaceControlPlaneAssignmentHandlerClone(handle C.uint64_t) C.uint64_t {
	val, ok := FfiConverterControlPlaneAssignmentHandlerINSTANCE.handleMap.tryGet(uint64(handle))
	if !ok {
		panic(fmt.Errorf("no callback in handle map: %d", handle))
	}
	return C.uint64_t(FfiConverterControlPlaneAssignmentHandlerINSTANCE.handleMap.insert(val))
}

func (c FfiConverterControlPlaneAssignmentHandler) register() {
	C.uniffi_blazen_uniffi_fn_init_callback_vtable_controlplaneassignmenthandler(&UniffiVTableCallbackInterfaceControlPlaneAssignmentHandlerINSTANCE)
}

// gRPC client for the orchestrator side of the control plane.
//
// Construct with [`ControlPlaneClient::connect`] (async) or
// [`ControlPlaneClient::connect_blocking`] (sync). All RPCs are
// serialised behind an inner [`tokio::sync::Mutex`] held inside the
// upstream [`CoreClient`]; concurrent calls on the same handle are safe
// but each method holds the mutex for the duration of its RPC.
type ControlPlaneClientInterface interface {
	// Cancel an in-flight run.
	//
	// # Errors
	//
	// Returns [`BlazenError::Validation`] if `run_id` is not a valid
	// UUID; [`BlazenError::Workflow`] for server-side errors.
	CancelWorkflow(runId string) (ControlPlaneRunStateSnapshot, error)
	// Look up the current state of a run.
	//
	// # Errors
	//
	// Returns [`BlazenError::Validation`] if `run_id` is not a valid
	// UUID; [`BlazenError::Workflow`] for server-side errors.
	DescribeWorkflow(runId string) (ControlPlaneRunStateSnapshot, error)
	// Tell the control plane to drain `node_id`.
	//
	// `immediate = true` asks the worker to stop now; `false` lets
	// it finish in-flight assignments before disconnecting.
	//
	// # Errors
	//
	// Returns [`BlazenError::ControlPlane`] for RPC failures.
	DrainWorker(nodeId string, immediate bool) error
	// List currently-connected workers.
	//
	// # Errors
	//
	// Returns [`BlazenError::Workflow`] for server-side errors.
	ListWorkers() ([]ControlPlaneWorkerInfo, error)
	// Submit a workflow to the control plane.
	//
	// Returns the initial [`ControlPlaneRunStateSnapshot`] (status will
	// usually be `Pending` or `Running` immediately after submission).
	//
	// # Errors
	//
	// Returns [`BlazenError::Validation`] if `request.input_json` is
	// not valid JSON; [`BlazenError::Workflow`] for server-side errors.
	SubmitWorkflow(request ControlPlaneSubmitRequest) (ControlPlaneRunStateSnapshot, error)
	// Subscribe to events for `run_id`, forwarding each event to
	// `subscriber` until the stream terminates.
	//
	// Returns a [`ControlPlaneSubscription`] handle; call
	// [`ControlPlaneSubscription::cancel`] to stop pumping events
	// before the run completes. The pump task always invokes either
	// `on_close` or `on_error` exactly once before exiting.
	//
	// # Errors
	//
	// Returns [`BlazenError::Validation`] if `run_id` is not a valid
	// UUID; [`BlazenError::Workflow`] if the server rejects the
	// subscription request itself.
	SubscribeRunEvents(runId string, subscriber ControlPlaneRunEventSubscriber) (*ControlPlaneSubscription, error)
}

// gRPC client for the orchestrator side of the control plane.
//
// Construct with [`ControlPlaneClient::connect`] (async) or
// [`ControlPlaneClient::connect_blocking`] (sync). All RPCs are
// serialised behind an inner [`tokio::sync::Mutex`] held inside the
// upstream [`CoreClient`]; concurrent calls on the same handle are safe
// but each method holds the mutex for the duration of its RPC.
type ControlPlaneClient struct {
	ffiObject FfiObject
}

// Async constructor. Use from Swift `async` / Kotlin `suspend`
// callers.
//
// # Errors
//
// Same as [`ControlPlaneClient::connect_blocking`].
func ControlPlaneClientConnect(endpoint string) (*ControlPlaneClient, error) {
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
			res := C.ffi_blazen_uniffi_rust_future_complete_u64(handle, status)
			return res
		},
		// liftFn
		func(ffi C.uint64_t) *ControlPlaneClient {
			return FfiConverterControlPlaneClientINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_constructor_controlplaneclient_connect(FfiConverterStringINSTANCE.Lower(endpoint)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_u64(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_u64(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// Synchronous constructor. Blocks the current thread on the shared
// Tokio runtime while the TCP/HTTP-2 handshake completes.
//
// # Errors
//
// Returns [`BlazenError::ControlPlane`] (`kind = "Transport"`) if
// the endpoint URI is invalid or the handshake fails.
func ControlPlaneClientConnectBlocking(endpoint string) (*ControlPlaneClient, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_constructor_controlplaneclient_connect_blocking(FfiConverterStringINSTANCE.Lower(endpoint), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *ControlPlaneClient
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterControlPlaneClientINSTANCE.Lift(_uniffiRV), nil
	}
}

// Cancel an in-flight run.
//
// # Errors
//
// Returns [`BlazenError::Validation`] if `run_id` is not a valid
// UUID; [`BlazenError::Workflow`] for server-side errors.
func (_self *ControlPlaneClient) CancelWorkflow(runId string) (ControlPlaneRunStateSnapshot, error) {
	_pointer := _self.ffiObject.incrementPointer("*ControlPlaneClient")
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
		func(ffi RustBufferI) ControlPlaneRunStateSnapshot {
			return FfiConverterControlPlaneRunStateSnapshotINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_controlplaneclient_cancel_workflow(
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

// Look up the current state of a run.
//
// # Errors
//
// Returns [`BlazenError::Validation`] if `run_id` is not a valid
// UUID; [`BlazenError::Workflow`] for server-side errors.
func (_self *ControlPlaneClient) DescribeWorkflow(runId string) (ControlPlaneRunStateSnapshot, error) {
	_pointer := _self.ffiObject.incrementPointer("*ControlPlaneClient")
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
		func(ffi RustBufferI) ControlPlaneRunStateSnapshot {
			return FfiConverterControlPlaneRunStateSnapshotINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_controlplaneclient_describe_workflow(
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

// Tell the control plane to drain `node_id`.
//
// `immediate = true` asks the worker to stop now; `false` lets
// it finish in-flight assignments before disconnecting.
//
// # Errors
//
// Returns [`BlazenError::ControlPlane`] for RPC failures.
func (_self *ControlPlaneClient) DrainWorker(nodeId string, immediate bool) error {
	_pointer := _self.ffiObject.incrementPointer("*ControlPlaneClient")
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
		C.uniffi_blazen_uniffi_fn_method_controlplaneclient_drain_worker(
			_pointer, FfiConverterStringINSTANCE.Lower(nodeId), FfiConverterBoolINSTANCE.Lower(immediate)),
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

// List currently-connected workers.
//
// # Errors
//
// Returns [`BlazenError::Workflow`] for server-side errors.
func (_self *ControlPlaneClient) ListWorkers() ([]ControlPlaneWorkerInfo, error) {
	_pointer := _self.ffiObject.incrementPointer("*ControlPlaneClient")
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
		func(ffi RustBufferI) []ControlPlaneWorkerInfo {
			return FfiConverterSequenceControlPlaneWorkerInfoINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_controlplaneclient_list_workers(
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

// Submit a workflow to the control plane.
//
// Returns the initial [`ControlPlaneRunStateSnapshot`] (status will
// usually be `Pending` or `Running` immediately after submission).
//
// # Errors
//
// Returns [`BlazenError::Validation`] if `request.input_json` is
// not valid JSON; [`BlazenError::Workflow`] for server-side errors.
func (_self *ControlPlaneClient) SubmitWorkflow(request ControlPlaneSubmitRequest) (ControlPlaneRunStateSnapshot, error) {
	_pointer := _self.ffiObject.incrementPointer("*ControlPlaneClient")
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
		func(ffi RustBufferI) ControlPlaneRunStateSnapshot {
			return FfiConverterControlPlaneRunStateSnapshotINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_controlplaneclient_submit_workflow(
			_pointer, FfiConverterControlPlaneSubmitRequestINSTANCE.Lower(request)),
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

// Subscribe to events for `run_id`, forwarding each event to
// `subscriber` until the stream terminates.
//
// Returns a [`ControlPlaneSubscription`] handle; call
// [`ControlPlaneSubscription::cancel`] to stop pumping events
// before the run completes. The pump task always invokes either
// `on_close` or `on_error` exactly once before exiting.
//
// # Errors
//
// Returns [`BlazenError::Validation`] if `run_id` is not a valid
// UUID; [`BlazenError::Workflow`] if the server rejects the
// subscription request itself.
func (_self *ControlPlaneClient) SubscribeRunEvents(runId string, subscriber ControlPlaneRunEventSubscriber) (*ControlPlaneSubscription, error) {
	_pointer := _self.ffiObject.incrementPointer("*ControlPlaneClient")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
			res := C.ffi_blazen_uniffi_rust_future_complete_u64(handle, status)
			return res
		},
		// liftFn
		func(ffi C.uint64_t) *ControlPlaneSubscription {
			return FfiConverterControlPlaneSubscriptionINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_controlplaneclient_subscribe_run_events(
			_pointer, FfiConverterStringINSTANCE.Lower(runId), FfiConverterControlPlaneRunEventSubscriberINSTANCE.Lower(subscriber)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_u64(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_u64(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}
func (object *ControlPlaneClient) Destroy() {
	runtime.SetFinalizer(object, nil)
	object.ffiObject.destroy()
}

type FfiConverterControlPlaneClient struct{}

var FfiConverterControlPlaneClientINSTANCE = FfiConverterControlPlaneClient{}

func (c FfiConverterControlPlaneClient) Lift(handle C.uint64_t) *ControlPlaneClient {
	result := &ControlPlaneClient{
		newFfiObject(
			handle,
			func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
				return C.uniffi_blazen_uniffi_fn_clone_controlplaneclient(handle, status)
			},
			func(handle C.uint64_t, status *C.RustCallStatus) {
				C.uniffi_blazen_uniffi_fn_free_controlplaneclient(handle, status)
			},
		),
	}
	runtime.SetFinalizer(result, (*ControlPlaneClient).Destroy)
	return result
}

func (c FfiConverterControlPlaneClient) Read(reader io.Reader) *ControlPlaneClient {
	return c.Lift(C.uint64_t(readUint64(reader)))
}

func (c FfiConverterControlPlaneClient) Lower(value *ControlPlaneClient) C.uint64_t {
	// SAFETY (audited 2026-05-13): incrementPointer calls cloneFunction
	// which does Arc::clone on the Rust side, bumping the Rust refcount
	// independently of the Go-side callCounter. The defer below only
	// decrements the (redundant) Go counter; the returned handle survives
	// because the C caller owns its own Arc refcount via Arc::from_raw.
	handle := value.ffiObject.incrementPointer("*ControlPlaneClient")
	defer value.ffiObject.decrementPointer()
	return handle
}

func (c FfiConverterControlPlaneClient) Write(writer io.Writer, value *ControlPlaneClient) {
	writeUint64(writer, uint64(c.Lower(value)))
}

func LiftFromExternalControlPlaneClient(handle uint64) *ControlPlaneClient {
	return FfiConverterControlPlaneClientINSTANCE.Lift(C.uint64_t(handle))
}

func LowerToExternalControlPlaneClient(value *ControlPlaneClient) uint64 {
	return uint64(FfiConverterControlPlaneClientINSTANCE.Lower(value))
}

type FfiDestroyerControlPlaneClient struct{}

func (_ FfiDestroyerControlPlaneClient) Destroy(value *ControlPlaneClient) {
	value.Destroy()
}

// Foreign-implementable subscriber that observes a per-run event stream
// opened by [`ControlPlaneClient::subscribe_run_events`].
//
// Like [`ControlPlaneAssignmentHandler`], every method is synchronous
// on the foreign side. The subscription pumps inbound events on the
// shared Tokio runtime and invokes the callbacks in the order they
// arrive; foreign callers wanting concurrent processing should spawn
// from inside `on_event`.
type ControlPlaneRunEventSubscriber interface {
	// One event arrived from the run.
	OnEvent(event ControlPlaneRunEvent)
	// Stream ended cleanly (the run reached a terminal state).
	OnClose()
	// Stream errored. `error` is best-effort and may not survive a
	// reconnect-then-retry cycle.
	OnError(error string)
}

// Foreign-implementable subscriber that observes a per-run event stream
// opened by [`ControlPlaneClient::subscribe_run_events`].
//
// Like [`ControlPlaneAssignmentHandler`], every method is synchronous
// on the foreign side. The subscription pumps inbound events on the
// shared Tokio runtime and invokes the callbacks in the order they
// arrive; foreign callers wanting concurrent processing should spawn
// from inside `on_event`.
type ControlPlaneRunEventSubscriberImpl struct {
	ffiObject FfiObject
}

// One event arrived from the run.
func (_self *ControlPlaneRunEventSubscriberImpl) OnEvent(event ControlPlaneRunEvent) {
	_pointer := _self.ffiObject.incrementPointer("ControlPlaneRunEventSubscriber")
	defer _self.ffiObject.decrementPointer()
	rustCall(func(_uniffiStatus *C.RustCallStatus) bool {
		C.uniffi_blazen_uniffi_fn_method_controlplaneruneventsubscriber_on_event(
			_pointer, FfiConverterControlPlaneRunEventINSTANCE.Lower(event), _uniffiStatus)
		return false
	})
}

// Stream ended cleanly (the run reached a terminal state).
func (_self *ControlPlaneRunEventSubscriberImpl) OnClose() {
	_pointer := _self.ffiObject.incrementPointer("ControlPlaneRunEventSubscriber")
	defer _self.ffiObject.decrementPointer()
	rustCall(func(_uniffiStatus *C.RustCallStatus) bool {
		C.uniffi_blazen_uniffi_fn_method_controlplaneruneventsubscriber_on_close(
			_pointer, _uniffiStatus)
		return false
	})
}

// Stream errored. `error` is best-effort and may not survive a
// reconnect-then-retry cycle.
func (_self *ControlPlaneRunEventSubscriberImpl) OnError(error string) {
	_pointer := _self.ffiObject.incrementPointer("ControlPlaneRunEventSubscriber")
	defer _self.ffiObject.decrementPointer()
	rustCall(func(_uniffiStatus *C.RustCallStatus) bool {
		C.uniffi_blazen_uniffi_fn_method_controlplaneruneventsubscriber_on_error(
			_pointer, FfiConverterStringINSTANCE.Lower(error), _uniffiStatus)
		return false
	})
}
func (object *ControlPlaneRunEventSubscriberImpl) Destroy() {
	runtime.SetFinalizer(object, nil)
	object.ffiObject.destroy()
}

type FfiConverterControlPlaneRunEventSubscriber struct {
	handleMap *concurrentHandleMap[ControlPlaneRunEventSubscriber]
}

var FfiConverterControlPlaneRunEventSubscriberINSTANCE = FfiConverterControlPlaneRunEventSubscriber{
	handleMap: newConcurrentHandleMap[ControlPlaneRunEventSubscriber](),
}

func (c FfiConverterControlPlaneRunEventSubscriber) Lift(handle C.uint64_t) ControlPlaneRunEventSubscriber {
	if uint64(handle)&1 == 0 {
		// Rust-generated handle (even), construct a new object wrapping the handle
		result := &ControlPlaneRunEventSubscriberImpl{
			newFfiObject(
				handle,
				func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
					return C.uniffi_blazen_uniffi_fn_clone_controlplaneruneventsubscriber(handle, status)
				},
				func(handle C.uint64_t, status *C.RustCallStatus) {
					C.uniffi_blazen_uniffi_fn_free_controlplaneruneventsubscriber(handle, status)
				},
			),
		}
		runtime.SetFinalizer(result, (*ControlPlaneRunEventSubscriberImpl).Destroy)
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

func (c FfiConverterControlPlaneRunEventSubscriber) Read(reader io.Reader) ControlPlaneRunEventSubscriber {
	return c.Lift(C.uint64_t(readUint64(reader)))
}

func (c FfiConverterControlPlaneRunEventSubscriber) Lower(value ControlPlaneRunEventSubscriber) C.uint64_t {
	// SAFETY (audited 2026-05-13): incrementPointer calls cloneFunction
	// which does Arc::clone on the Rust side, bumping the Rust refcount
	// independently of the Go-side callCounter. The defer below only
	// decrements the (redundant) Go counter; the returned handle survives
	// because the C caller owns its own Arc refcount via Arc::from_raw.
	if val, ok := value.(*ControlPlaneRunEventSubscriberImpl); ok {
		// Rust-backed object, clone the handle
		handle := val.ffiObject.incrementPointer("ControlPlaneRunEventSubscriber")
		defer val.ffiObject.decrementPointer()
		return handle
	} else {
		// Go-backed object, insert into handle map
		return C.uint64_t(c.handleMap.insert(value))
	}
}

func (c FfiConverterControlPlaneRunEventSubscriber) Write(writer io.Writer, value ControlPlaneRunEventSubscriber) {
	writeUint64(writer, uint64(c.Lower(value)))
}

func LiftFromExternalControlPlaneRunEventSubscriber(handle uint64) ControlPlaneRunEventSubscriber {
	return FfiConverterControlPlaneRunEventSubscriberINSTANCE.Lift(C.uint64_t(handle))
}

func LowerToExternalControlPlaneRunEventSubscriber(value ControlPlaneRunEventSubscriber) uint64 {
	return uint64(FfiConverterControlPlaneRunEventSubscriberINSTANCE.Lower(value))
}

type FfiDestroyerControlPlaneRunEventSubscriber struct{}

func (_ FfiDestroyerControlPlaneRunEventSubscriber) Destroy(value ControlPlaneRunEventSubscriber) {
	if val, ok := value.(*ControlPlaneRunEventSubscriberImpl); ok {
		val.Destroy()
	}
}

//export blazen_uniffi_controlplane_cgo_dispatchCallbackInterfaceControlPlaneRunEventSubscriberMethod0
func blazen_uniffi_controlplane_cgo_dispatchCallbackInterfaceControlPlaneRunEventSubscriberMethod0(uniffiHandle C.uint64_t, event C.RustBuffer, uniffiOutReturn *C.void, callStatus *C.RustCallStatus) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterControlPlaneRunEventSubscriberINSTANCE.handleMap.tryGet(handle)
	if !ok {
		panic(fmt.Errorf("no callback in handle map: %d", handle))
	}

	uniffiObj.OnEvent(
		FfiConverterControlPlaneRunEventINSTANCE.Lift(GoRustBuffer{
			inner: event,
		}),
	)

}

//export blazen_uniffi_controlplane_cgo_dispatchCallbackInterfaceControlPlaneRunEventSubscriberMethod1
func blazen_uniffi_controlplane_cgo_dispatchCallbackInterfaceControlPlaneRunEventSubscriberMethod1(uniffiHandle C.uint64_t, uniffiOutReturn *C.void, callStatus *C.RustCallStatus) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterControlPlaneRunEventSubscriberINSTANCE.handleMap.tryGet(handle)
	if !ok {
		panic(fmt.Errorf("no callback in handle map: %d", handle))
	}

	uniffiObj.OnClose()

}

//export blazen_uniffi_controlplane_cgo_dispatchCallbackInterfaceControlPlaneRunEventSubscriberMethod2
func blazen_uniffi_controlplane_cgo_dispatchCallbackInterfaceControlPlaneRunEventSubscriberMethod2(uniffiHandle C.uint64_t, error C.RustBuffer, uniffiOutReturn *C.void, callStatus *C.RustCallStatus) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterControlPlaneRunEventSubscriberINSTANCE.handleMap.tryGet(handle)
	if !ok {
		panic(fmt.Errorf("no callback in handle map: %d", handle))
	}

	uniffiObj.OnError(
		FfiConverterStringINSTANCE.Lift(GoRustBuffer{
			inner: error,
		}),
	)

}

var UniffiVTableCallbackInterfaceControlPlaneRunEventSubscriberINSTANCE = C.UniffiVTableCallbackInterfaceControlPlaneRunEventSubscriber{
	uniffiFree:  (C.UniffiCallbackInterfaceFree)(C.blazen_uniffi_controlplane_cgo_dispatchCallbackInterfaceControlPlaneRunEventSubscriberFree),
	uniffiClone: (C.UniffiCallbackInterfaceClone)(C.blazen_uniffi_controlplane_cgo_dispatchCallbackInterfaceControlPlaneRunEventSubscriberClone),
	onEvent:     (C.UniffiCallbackInterfaceControlPlaneRunEventSubscriberMethod0)(C.blazen_uniffi_controlplane_cgo_dispatchCallbackInterfaceControlPlaneRunEventSubscriberMethod0),
	onClose:     (C.UniffiCallbackInterfaceControlPlaneRunEventSubscriberMethod1)(C.blazen_uniffi_controlplane_cgo_dispatchCallbackInterfaceControlPlaneRunEventSubscriberMethod1),
	onError:     (C.UniffiCallbackInterfaceControlPlaneRunEventSubscriberMethod2)(C.blazen_uniffi_controlplane_cgo_dispatchCallbackInterfaceControlPlaneRunEventSubscriberMethod2),
}

//export blazen_uniffi_controlplane_cgo_dispatchCallbackInterfaceControlPlaneRunEventSubscriberFree
func blazen_uniffi_controlplane_cgo_dispatchCallbackInterfaceControlPlaneRunEventSubscriberFree(handle C.uint64_t) {
	FfiConverterControlPlaneRunEventSubscriberINSTANCE.handleMap.remove(uint64(handle))
}

//export blazen_uniffi_controlplane_cgo_dispatchCallbackInterfaceControlPlaneRunEventSubscriberClone
func blazen_uniffi_controlplane_cgo_dispatchCallbackInterfaceControlPlaneRunEventSubscriberClone(handle C.uint64_t) C.uint64_t {
	val, ok := FfiConverterControlPlaneRunEventSubscriberINSTANCE.handleMap.tryGet(uint64(handle))
	if !ok {
		panic(fmt.Errorf("no callback in handle map: %d", handle))
	}
	return C.uint64_t(FfiConverterControlPlaneRunEventSubscriberINSTANCE.handleMap.insert(val))
}

func (c FfiConverterControlPlaneRunEventSubscriber) register() {
	C.uniffi_blazen_uniffi_fn_init_callback_vtable_controlplaneruneventsubscriber(&UniffiVTableCallbackInterfaceControlPlaneRunEventSubscriberINSTANCE)
}

// Handle to an active run-event subscription. Drop the handle or call
// [`ControlPlaneSubscription::cancel`] to stop pumping events.
type ControlPlaneSubscriptionInterface interface {
	// Cancel the subscription. Idempotent. After cancellation, the
	// subscriber's `on_close` fires (best-effort) before the pump task
	// exits.
	Cancel()
}

// Handle to an active run-event subscription. Drop the handle or call
// [`ControlPlaneSubscription::cancel`] to stop pumping events.
type ControlPlaneSubscription struct {
	ffiObject FfiObject
}

// Cancel the subscription. Idempotent. After cancellation, the
// subscriber's `on_close` fires (best-effort) before the pump task
// exits.
func (_self *ControlPlaneSubscription) Cancel() {
	_pointer := _self.ffiObject.incrementPointer("*ControlPlaneSubscription")
	defer _self.ffiObject.decrementPointer()
	rustCall(func(_uniffiStatus *C.RustCallStatus) bool {
		C.uniffi_blazen_uniffi_fn_method_controlplanesubscription_cancel(
			_pointer, _uniffiStatus)
		return false
	})
}
func (object *ControlPlaneSubscription) Destroy() {
	runtime.SetFinalizer(object, nil)
	object.ffiObject.destroy()
}

type FfiConverterControlPlaneSubscription struct{}

var FfiConverterControlPlaneSubscriptionINSTANCE = FfiConverterControlPlaneSubscription{}

func (c FfiConverterControlPlaneSubscription) Lift(handle C.uint64_t) *ControlPlaneSubscription {
	result := &ControlPlaneSubscription{
		newFfiObject(
			handle,
			func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
				return C.uniffi_blazen_uniffi_fn_clone_controlplanesubscription(handle, status)
			},
			func(handle C.uint64_t, status *C.RustCallStatus) {
				C.uniffi_blazen_uniffi_fn_free_controlplanesubscription(handle, status)
			},
		),
	}
	runtime.SetFinalizer(result, (*ControlPlaneSubscription).Destroy)
	return result
}

func (c FfiConverterControlPlaneSubscription) Read(reader io.Reader) *ControlPlaneSubscription {
	return c.Lift(C.uint64_t(readUint64(reader)))
}

func (c FfiConverterControlPlaneSubscription) Lower(value *ControlPlaneSubscription) C.uint64_t {
	// SAFETY (audited 2026-05-13): incrementPointer calls cloneFunction
	// which does Arc::clone on the Rust side, bumping the Rust refcount
	// independently of the Go-side callCounter. The defer below only
	// decrements the (redundant) Go counter; the returned handle survives
	// because the C caller owns its own Arc refcount via Arc::from_raw.
	handle := value.ffiObject.incrementPointer("*ControlPlaneSubscription")
	defer value.ffiObject.decrementPointer()
	return handle
}

func (c FfiConverterControlPlaneSubscription) Write(writer io.Writer, value *ControlPlaneSubscription) {
	writeUint64(writer, uint64(c.Lower(value)))
}

func LiftFromExternalControlPlaneSubscription(handle uint64) *ControlPlaneSubscription {
	return FfiConverterControlPlaneSubscriptionINSTANCE.Lift(C.uint64_t(handle))
}

func LowerToExternalControlPlaneSubscription(value *ControlPlaneSubscription) uint64 {
	return uint64(FfiConverterControlPlaneSubscriptionINSTANCE.Lower(value))
}

type FfiDestroyerControlPlaneSubscription struct{}

func (_ FfiDestroyerControlPlaneSubscription) Destroy(value *ControlPlaneSubscription) {
	value.Destroy()
}

// gRPC worker-side handle for the control plane.
//
// Wraps [`CoreWorker`] behind an `Arc<Mutex<Option<...>>>` because
// upstream [`CoreWorker::run`] consumes `self` by value. The first
// successful call to [`ControlPlaneWorker::run`] takes the worker out
// of the mutex; subsequent calls fail with [`BlazenError::Validation`].
// [`ControlPlaneWorker::shutdown`] is exposed separately because it
// needs to fire even while `run` is in flight.
type ControlPlaneWorkerInterface interface {
	// Drive the worker session forever (or until shutdown / drain /
	// retry exhaustion).
	//
	// Adapts `handler` to the upstream
	// [`blazen_controlplane::AssignmentHandler`] trait and hands it to
	// [`CoreWorker::run`]. Consumes the underlying worker — calling
	// `run` twice on the same handle returns
	// [`BlazenError::Validation`].
	//
	// # Errors
	//
	// Returns [`BlazenError::ControlPlane`] for transport / retry
	// failures, or [`BlazenError::Validation`] if `run` is called more
	// than once.
	Run(handler ControlPlaneAssignmentHandler) error
	// Signal the worker to stop. Returns immediately; any in-flight
	// [`ControlPlaneWorker::run`] call will return cleanly once the
	// in-flight assignments have been told to cancel.
	//
	// Idempotent.
	Shutdown()
}

// gRPC worker-side handle for the control plane.
//
// Wraps [`CoreWorker`] behind an `Arc<Mutex<Option<...>>>` because
// upstream [`CoreWorker::run`] consumes `self` by value. The first
// successful call to [`ControlPlaneWorker::run`] takes the worker out
// of the mutex; subsequent calls fail with [`BlazenError::Validation`].
// [`ControlPlaneWorker::shutdown`] is exposed separately because it
// needs to fire even while `run` is in flight.
type ControlPlaneWorker struct {
	ffiObject FfiObject
}

// Synchronous constructor.
//
// Builds a [`WorkerConfig`] with `Fixed { max_in_flight: 1 }`
// admission and the supplied `capabilities`, validates the endpoint
// URI, and returns a worker that has *not* yet opened the bidi
// stream — call [`ControlPlaneWorker::run`] to do that.
//
// # Errors
//
// Returns [`BlazenError::ControlPlane`] (`kind = "Transport"`) if
// `endpoint` cannot be parsed as a URI.
func ControlPlaneWorkerNewBlocking(endpoint string, nodeId string, capabilities []ControlPlaneWorkerCapability) (*ControlPlaneWorker, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_constructor_controlplaneworker_new_blocking(FfiConverterStringINSTANCE.Lower(endpoint), FfiConverterStringINSTANCE.Lower(nodeId), FfiConverterSequenceControlPlaneWorkerCapabilityINSTANCE.Lower(capabilities), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *ControlPlaneWorker
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterControlPlaneWorkerINSTANCE.Lift(_uniffiRV), nil
	}
}

// Drive the worker session forever (or until shutdown / drain /
// retry exhaustion).
//
// Adapts `handler` to the upstream
// [`blazen_controlplane::AssignmentHandler`] trait and hands it to
// [`CoreWorker::run`]. Consumes the underlying worker — calling
// `run` twice on the same handle returns
// [`BlazenError::Validation`].
//
// # Errors
//
// Returns [`BlazenError::ControlPlane`] for transport / retry
// failures, or [`BlazenError::Validation`] if `run` is called more
// than once.
func (_self *ControlPlaneWorker) Run(handler ControlPlaneAssignmentHandler) error {
	_pointer := _self.ffiObject.incrementPointer("*ControlPlaneWorker")
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
		C.uniffi_blazen_uniffi_fn_method_controlplaneworker_run(
			_pointer, FfiConverterControlPlaneAssignmentHandlerINSTANCE.Lower(handler)),
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

// Signal the worker to stop. Returns immediately; any in-flight
// [`ControlPlaneWorker::run`] call will return cleanly once the
// in-flight assignments have been told to cancel.
//
// Idempotent.
func (_self *ControlPlaneWorker) Shutdown() {
	_pointer := _self.ffiObject.incrementPointer("*ControlPlaneWorker")
	defer _self.ffiObject.decrementPointer()
	rustCall(func(_uniffiStatus *C.RustCallStatus) bool {
		C.uniffi_blazen_uniffi_fn_method_controlplaneworker_shutdown(
			_pointer, _uniffiStatus)
		return false
	})
}
func (object *ControlPlaneWorker) Destroy() {
	runtime.SetFinalizer(object, nil)
	object.ffiObject.destroy()
}

type FfiConverterControlPlaneWorker struct{}

var FfiConverterControlPlaneWorkerINSTANCE = FfiConverterControlPlaneWorker{}

func (c FfiConverterControlPlaneWorker) Lift(handle C.uint64_t) *ControlPlaneWorker {
	result := &ControlPlaneWorker{
		newFfiObject(
			handle,
			func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
				return C.uniffi_blazen_uniffi_fn_clone_controlplaneworker(handle, status)
			},
			func(handle C.uint64_t, status *C.RustCallStatus) {
				C.uniffi_blazen_uniffi_fn_free_controlplaneworker(handle, status)
			},
		),
	}
	runtime.SetFinalizer(result, (*ControlPlaneWorker).Destroy)
	return result
}

func (c FfiConverterControlPlaneWorker) Read(reader io.Reader) *ControlPlaneWorker {
	return c.Lift(C.uint64_t(readUint64(reader)))
}

func (c FfiConverterControlPlaneWorker) Lower(value *ControlPlaneWorker) C.uint64_t {
	// SAFETY (audited 2026-05-13): incrementPointer calls cloneFunction
	// which does Arc::clone on the Rust side, bumping the Rust refcount
	// independently of the Go-side callCounter. The defer below only
	// decrements the (redundant) Go counter; the returned handle survives
	// because the C caller owns its own Arc refcount via Arc::from_raw.
	handle := value.ffiObject.incrementPointer("*ControlPlaneWorker")
	defer value.ffiObject.decrementPointer()
	return handle
}

func (c FfiConverterControlPlaneWorker) Write(writer io.Writer, value *ControlPlaneWorker) {
	writeUint64(writer, uint64(c.Lower(value)))
}

func LiftFromExternalControlPlaneWorker(handle uint64) *ControlPlaneWorker {
	return FfiConverterControlPlaneWorkerINSTANCE.Lift(C.uint64_t(handle))
}

func LowerToExternalControlPlaneWorker(value *ControlPlaneWorker) uint64 {
	return uint64(FfiConverterControlPlaneWorkerINSTANCE.Lower(value))
}

type FfiDestroyerControlPlaneWorker struct{}

func (_ FfiDestroyerControlPlaneWorker) Destroy(value *ControlPlaneWorker) {
	value.Destroy()
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
	Complete(request ModelRequest) (ModelResponse, error)
	// Perform a streaming chat completion, pushing chunks into the supplied
	// sink. The implementation must call `sink.on_done` exactly once on
	// success or `sink.on_error` exactly once on failure.
	Stream(request ModelRequest, sink CompletionStreamSink) error
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
func (_self *CustomProviderImpl) Complete(request ModelRequest) (ModelResponse, error) {
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
		func(ffi RustBufferI) ModelResponse {
			return FfiConverterModelResponseINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_customprovider_complete(
			_pointer, FfiConverterModelRequestINSTANCE.Lower(request)),
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
func (_self *CustomProviderImpl) Stream(request ModelRequest, sink CompletionStreamSink) error {
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
			_pointer, FfiConverterModelRequestINSTANCE.Lower(request), FfiConverterCompletionStreamSinkINSTANCE.Lower(sink)),
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
	// SAFETY (audited 2026-05-13): incrementPointer calls cloneFunction
	// which does Arc::clone on the Rust side, bumping the Rust refcount
	// independently of the Go-side callCounter. The defer below only
	// decrements the (redundant) Go counter; the returned handle survives
	// because the C caller owns its own Arc refcount via Arc::from_raw.
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
				FfiConverterModelRequestINSTANCE.Lift(GoRustBuffer{
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

		*uniffiOutReturn = FfiConverterModelResponseINSTANCE.Lower(res)
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
				FfiConverterModelRequestINSTANCE.Lift(GoRustBuffer{
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
	// expecting an opaque `Model`-shaped handle.
	AsBase() *BaseProvider
	// Clone a voice from reference audio.
	CloneVoice(request VoiceCloneRequest) (VoiceHandle, error)
	// Perform a non-streaming chat completion. Applies any configured
	// completion defaults (system prompt, tools, response format) before
	// dispatching to the inner provider.
	Complete(request ModelRequest) (ModelResponse, error)
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
	Stream(request ModelRequest, sink CompletionStreamSink) error
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
// expecting an opaque `Model`-shaped handle.
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
func (_self *CustomProviderHandle) Complete(request ModelRequest) (ModelResponse, error) {
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
		func(ffi RustBufferI) ModelResponse {
			return FfiConverterModelResponseINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_customproviderhandle_complete(
			_pointer, FfiConverterModelRequestINSTANCE.Lower(request)),
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
func (_self *CustomProviderHandle) Stream(request ModelRequest, sink CompletionStreamSink) error {
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
			_pointer, FfiConverterModelRequestINSTANCE.Lower(request), FfiConverterCompletionStreamSinkINSTANCE.Lower(sink)),
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
	// SAFETY (audited 2026-05-13): incrementPointer calls cloneFunction
	// which does Arc::clone on the Rust side, bumping the Rust refcount
	// independently of the Go-side callCounter. The defer below only
	// decrements the (redundant) Go counter; the returned handle survives
	// because the C caller owns its own Arc refcount via Arc::from_raw.
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
	// SAFETY (audited 2026-05-13): incrementPointer calls cloneFunction
	// which does Arc::clone on the Rust side, bumping the Rust refcount
	// independently of the Go-side callCounter. The defer below only
	// decrements the (redundant) Go counter; the returned handle survives
	// because the C caller owns its own Arc refcount via Arc::from_raw.
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

// Foreign-language implementation of a local (on-device) model.
//
// Implementors mirror the upstream [`blazen_llm::LocalModel`] trait but in
// FFI-friendly form: paths are `String`, the `device()` accessor returns a
// `String` ("cpu", "cuda:0", "metal", ...) that gets parsed back into
// [`blazen_llm::Device`] when forwarded to the manager.
//
// `is_loaded`, `memory_bytes`, `device`, `load_adapter`, `unload_adapter`,
// and `list_adapters` are NOT optional on this trait — UniFFI does not have a
// concept of "default trait method" that the foreign side can opt out of.
// Foreign implementors that don't care about a verb should return a sensible
// neutral value (`false` for `is_loaded`, `0` / `None` for `memory_bytes`,
// `"cpu"` for `device`, an empty `list_adapters`, or raise
// [`BlazenError::Unsupported`] from the adapter verbs).
type ForeignLocalModel interface {
	Load() error
	Unload() error
	IsLoaded() bool
	Device() string
	MemoryBytes() *uint64
	LoadAdapter(adapterDir string, options AdapterOptionsRecord) (AdapterHandleRecord, error)
	UnloadAdapter(handle AdapterHandleRecord) error
	ListAdapters() []AdapterStatusRecord
}

// Foreign-language implementation of a local (on-device) model.
//
// Implementors mirror the upstream [`blazen_llm::LocalModel`] trait but in
// FFI-friendly form: paths are `String`, the `device()` accessor returns a
// `String` ("cpu", "cuda:0", "metal", ...) that gets parsed back into
// [`blazen_llm::Device`] when forwarded to the manager.
//
// `is_loaded`, `memory_bytes`, `device`, `load_adapter`, `unload_adapter`,
// and `list_adapters` are NOT optional on this trait — UniFFI does not have a
// concept of "default trait method" that the foreign side can opt out of.
// Foreign implementors that don't care about a verb should return a sensible
// neutral value (`false` for `is_loaded`, `0` / `None` for `memory_bytes`,
// `"cpu"` for `device`, an empty `list_adapters`, or raise
// [`BlazenError::Unsupported`] from the adapter verbs).
type ForeignLocalModelImpl struct {
	ffiObject FfiObject
}

func (_self *ForeignLocalModelImpl) Load() error {
	_pointer := _self.ffiObject.incrementPointer("ForeignLocalModel")
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
		C.uniffi_blazen_uniffi_fn_method_foreignlocalmodel_load(
			_pointer),
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

func (_self *ForeignLocalModelImpl) Unload() error {
	_pointer := _self.ffiObject.incrementPointer("ForeignLocalModel")
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
		C.uniffi_blazen_uniffi_fn_method_foreignlocalmodel_unload(
			_pointer),
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

func (_self *ForeignLocalModelImpl) IsLoaded() bool {
	_pointer := _self.ffiObject.incrementPointer("ForeignLocalModel")
	defer _self.ffiObject.decrementPointer()
	res, _ := uniffiRustCallAsync[error](
		nil,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) C.int8_t {
			res := C.ffi_blazen_uniffi_rust_future_complete_i8(handle, status)
			return res
		},
		// liftFn
		func(ffi C.int8_t) bool {
			return FfiConverterBoolINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_foreignlocalmodel_is_loaded(
			_pointer),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_i8(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_i8(handle)
		},
	)

	return res
}

func (_self *ForeignLocalModelImpl) Device() string {
	_pointer := _self.ffiObject.incrementPointer("ForeignLocalModel")
	defer _self.ffiObject.decrementPointer()
	return FfiConverterStringINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) RustBufferI {
		return GoRustBuffer{
			inner: C.uniffi_blazen_uniffi_fn_method_foreignlocalmodel_device(
				_pointer, _uniffiStatus),
		}
	}))
}

func (_self *ForeignLocalModelImpl) MemoryBytes() *uint64 {
	_pointer := _self.ffiObject.incrementPointer("ForeignLocalModel")
	defer _self.ffiObject.decrementPointer()
	res, _ := uniffiRustCallAsync[error](
		nil,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) *uint64 {
			return FfiConverterOptionalUint64INSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_foreignlocalmodel_memory_bytes(
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

	return res
}

func (_self *ForeignLocalModelImpl) LoadAdapter(adapterDir string, options AdapterOptionsRecord) (AdapterHandleRecord, error) {
	_pointer := _self.ffiObject.incrementPointer("ForeignLocalModel")
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
		func(ffi RustBufferI) AdapterHandleRecord {
			return FfiConverterAdapterHandleRecordINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_foreignlocalmodel_load_adapter(
			_pointer, FfiConverterStringINSTANCE.Lower(adapterDir), FfiConverterAdapterOptionsRecordINSTANCE.Lower(options)),
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

func (_self *ForeignLocalModelImpl) UnloadAdapter(handle AdapterHandleRecord) error {
	_pointer := _self.ffiObject.incrementPointer("ForeignLocalModel")
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
		C.uniffi_blazen_uniffi_fn_method_foreignlocalmodel_unload_adapter(
			_pointer, FfiConverterAdapterHandleRecordINSTANCE.Lower(handle)),
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

func (_self *ForeignLocalModelImpl) ListAdapters() []AdapterStatusRecord {
	_pointer := _self.ffiObject.incrementPointer("ForeignLocalModel")
	defer _self.ffiObject.decrementPointer()
	res, _ := uniffiRustCallAsync[error](
		nil,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) []AdapterStatusRecord {
			return FfiConverterSequenceAdapterStatusRecordINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_foreignlocalmodel_list_adapters(
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

	return res
}
func (object *ForeignLocalModelImpl) Destroy() {
	runtime.SetFinalizer(object, nil)
	object.ffiObject.destroy()
}

type FfiConverterForeignLocalModel struct {
	handleMap *concurrentHandleMap[ForeignLocalModel]
}

var FfiConverterForeignLocalModelINSTANCE = FfiConverterForeignLocalModel{
	handleMap: newConcurrentHandleMap[ForeignLocalModel](),
}

func (c FfiConverterForeignLocalModel) Lift(handle C.uint64_t) ForeignLocalModel {
	if uint64(handle)&1 == 0 {
		// Rust-generated handle (even), construct a new object wrapping the handle
		result := &ForeignLocalModelImpl{
			newFfiObject(
				handle,
				func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
					return C.uniffi_blazen_uniffi_fn_clone_foreignlocalmodel(handle, status)
				},
				func(handle C.uint64_t, status *C.RustCallStatus) {
					C.uniffi_blazen_uniffi_fn_free_foreignlocalmodel(handle, status)
				},
			),
		}
		runtime.SetFinalizer(result, (*ForeignLocalModelImpl).Destroy)
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

func (c FfiConverterForeignLocalModel) Read(reader io.Reader) ForeignLocalModel {
	return c.Lift(C.uint64_t(readUint64(reader)))
}

func (c FfiConverterForeignLocalModel) Lower(value ForeignLocalModel) C.uint64_t {
	// SAFETY (audited 2026-05-13): incrementPointer calls cloneFunction
	// which does Arc::clone on the Rust side, bumping the Rust refcount
	// independently of the Go-side callCounter. The defer below only
	// decrements the (redundant) Go counter; the returned handle survives
	// because the C caller owns its own Arc refcount via Arc::from_raw.
	if val, ok := value.(*ForeignLocalModelImpl); ok {
		// Rust-backed object, clone the handle
		handle := val.ffiObject.incrementPointer("ForeignLocalModel")
		defer val.ffiObject.decrementPointer()
		return handle
	} else {
		// Go-backed object, insert into handle map
		return C.uint64_t(c.handleMap.insert(value))
	}
}

func (c FfiConverterForeignLocalModel) Write(writer io.Writer, value ForeignLocalModel) {
	writeUint64(writer, uint64(c.Lower(value)))
}

func LiftFromExternalForeignLocalModel(handle uint64) ForeignLocalModel {
	return FfiConverterForeignLocalModelINSTANCE.Lift(C.uint64_t(handle))
}

func LowerToExternalForeignLocalModel(value ForeignLocalModel) uint64 {
	return uint64(FfiConverterForeignLocalModelINSTANCE.Lower(value))
}

type FfiDestroyerForeignLocalModel struct{}

func (_ FfiDestroyerForeignLocalModel) Destroy(value ForeignLocalModel) {
	if val, ok := value.(*ForeignLocalModelImpl); ok {
		val.Destroy()
	}
}

//export blazen_uniffi_local_model_cgo_dispatchCallbackInterfaceForeignLocalModelMethod0
func blazen_uniffi_local_model_cgo_dispatchCallbackInterfaceForeignLocalModelMethod0(uniffiHandle C.uint64_t, uniffiFutureCallback C.UniffiForeignFutureCompleteVoid, uniffiCallbackData C.uint64_t, uniffiOutDroppedCallback *C.UniffiForeignFutureDroppedCallbackStruct) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterForeignLocalModelINSTANCE.handleMap.tryGet(handle)
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
			uniffiObj.Load()

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

//export blazen_uniffi_local_model_cgo_dispatchCallbackInterfaceForeignLocalModelMethod1
func blazen_uniffi_local_model_cgo_dispatchCallbackInterfaceForeignLocalModelMethod1(uniffiHandle C.uint64_t, uniffiFutureCallback C.UniffiForeignFutureCompleteVoid, uniffiCallbackData C.uint64_t, uniffiOutDroppedCallback *C.UniffiForeignFutureDroppedCallbackStruct) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterForeignLocalModelINSTANCE.handleMap.tryGet(handle)
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
			uniffiObj.Unload()

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

//export blazen_uniffi_local_model_cgo_dispatchCallbackInterfaceForeignLocalModelMethod2
func blazen_uniffi_local_model_cgo_dispatchCallbackInterfaceForeignLocalModelMethod2(uniffiHandle C.uint64_t, uniffiFutureCallback C.UniffiForeignFutureCompleteI8, uniffiCallbackData C.uint64_t, uniffiOutDroppedCallback *C.UniffiForeignFutureDroppedCallbackStruct) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterForeignLocalModelINSTANCE.handleMap.tryGet(handle)
	if !ok {
		panic(fmt.Errorf("no callback in handle map: %d", handle))
	}

	result := make(chan C.UniffiForeignFutureResultI8, 1)
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
			C.call_UniffiForeignFutureCompleteI8(uniffiFutureCallback, uniffiCallbackData, res)
		}
	}()

	// Eval callback asynchroniously
	go func() {
		asyncResult := &C.UniffiForeignFutureResultI8{}
		uniffiOutReturn := &asyncResult.returnValue
		defer func() {
			result <- *asyncResult
		}()

		res :=
			uniffiObj.IsLoaded()

		*uniffiOutReturn = FfiConverterBoolINSTANCE.Lower(res)
	}()
}

//export blazen_uniffi_local_model_cgo_dispatchCallbackInterfaceForeignLocalModelMethod3
func blazen_uniffi_local_model_cgo_dispatchCallbackInterfaceForeignLocalModelMethod3(uniffiHandle C.uint64_t, uniffiOutReturn *C.RustBuffer, callStatus *C.RustCallStatus) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterForeignLocalModelINSTANCE.handleMap.tryGet(handle)
	if !ok {
		panic(fmt.Errorf("no callback in handle map: %d", handle))
	}

	res :=
		uniffiObj.Device()

	*uniffiOutReturn = FfiConverterStringINSTANCE.Lower(res)
}

//export blazen_uniffi_local_model_cgo_dispatchCallbackInterfaceForeignLocalModelMethod4
func blazen_uniffi_local_model_cgo_dispatchCallbackInterfaceForeignLocalModelMethod4(uniffiHandle C.uint64_t, uniffiFutureCallback C.UniffiForeignFutureCompleteRustBuffer, uniffiCallbackData C.uint64_t, uniffiOutDroppedCallback *C.UniffiForeignFutureDroppedCallbackStruct) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterForeignLocalModelINSTANCE.handleMap.tryGet(handle)
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
		defer func() {
			result <- *asyncResult
		}()

		res :=
			uniffiObj.MemoryBytes()

		*uniffiOutReturn = FfiConverterOptionalUint64INSTANCE.Lower(res)
	}()
}

//export blazen_uniffi_local_model_cgo_dispatchCallbackInterfaceForeignLocalModelMethod5
func blazen_uniffi_local_model_cgo_dispatchCallbackInterfaceForeignLocalModelMethod5(uniffiHandle C.uint64_t, adapterDir C.RustBuffer, options C.RustBuffer, uniffiFutureCallback C.UniffiForeignFutureCompleteRustBuffer, uniffiCallbackData C.uint64_t, uniffiOutDroppedCallback *C.UniffiForeignFutureDroppedCallbackStruct) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterForeignLocalModelINSTANCE.handleMap.tryGet(handle)
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
			uniffiObj.LoadAdapter(
				FfiConverterStringINSTANCE.Lift(GoRustBuffer{
					inner: adapterDir,
				}),
				FfiConverterAdapterOptionsRecordINSTANCE.Lift(GoRustBuffer{
					inner: options,
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

		*uniffiOutReturn = FfiConverterAdapterHandleRecordINSTANCE.Lower(res)
	}()
}

//export blazen_uniffi_local_model_cgo_dispatchCallbackInterfaceForeignLocalModelMethod6
func blazen_uniffi_local_model_cgo_dispatchCallbackInterfaceForeignLocalModelMethod6(uniffiHandle C.uint64_t, adapterHandleBuf C.RustBuffer, uniffiFutureCallback C.UniffiForeignFutureCompleteVoid, uniffiCallbackData C.uint64_t, uniffiOutDroppedCallback *C.UniffiForeignFutureDroppedCallbackStruct) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterForeignLocalModelINSTANCE.handleMap.tryGet(handle)
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
			uniffiObj.UnloadAdapter(
				FfiConverterAdapterHandleRecordINSTANCE.Lift(GoRustBuffer{
					inner: adapterHandleBuf,
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

//export blazen_uniffi_local_model_cgo_dispatchCallbackInterfaceForeignLocalModelMethod7
func blazen_uniffi_local_model_cgo_dispatchCallbackInterfaceForeignLocalModelMethod7(uniffiHandle C.uint64_t, uniffiFutureCallback C.UniffiForeignFutureCompleteRustBuffer, uniffiCallbackData C.uint64_t, uniffiOutDroppedCallback *C.UniffiForeignFutureDroppedCallbackStruct) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterForeignLocalModelINSTANCE.handleMap.tryGet(handle)
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
		defer func() {
			result <- *asyncResult
		}()

		res :=
			uniffiObj.ListAdapters()

		*uniffiOutReturn = FfiConverterSequenceAdapterStatusRecordINSTANCE.Lower(res)
	}()
}

var UniffiVTableCallbackInterfaceForeignLocalModelINSTANCE = C.UniffiVTableCallbackInterfaceForeignLocalModel{
	uniffiFree:    (C.UniffiCallbackInterfaceFree)(C.blazen_uniffi_local_model_cgo_dispatchCallbackInterfaceForeignLocalModelFree),
	uniffiClone:   (C.UniffiCallbackInterfaceClone)(C.blazen_uniffi_local_model_cgo_dispatchCallbackInterfaceForeignLocalModelClone),
	load:          (C.UniffiCallbackInterfaceForeignLocalModelMethod0)(C.blazen_uniffi_local_model_cgo_dispatchCallbackInterfaceForeignLocalModelMethod0),
	unload:        (C.UniffiCallbackInterfaceForeignLocalModelMethod1)(C.blazen_uniffi_local_model_cgo_dispatchCallbackInterfaceForeignLocalModelMethod1),
	isLoaded:      (C.UniffiCallbackInterfaceForeignLocalModelMethod2)(C.blazen_uniffi_local_model_cgo_dispatchCallbackInterfaceForeignLocalModelMethod2),
	device:        (C.UniffiCallbackInterfaceForeignLocalModelMethod3)(C.blazen_uniffi_local_model_cgo_dispatchCallbackInterfaceForeignLocalModelMethod3),
	memoryBytes:   (C.UniffiCallbackInterfaceForeignLocalModelMethod4)(C.blazen_uniffi_local_model_cgo_dispatchCallbackInterfaceForeignLocalModelMethod4),
	loadAdapter:   (C.UniffiCallbackInterfaceForeignLocalModelMethod5)(C.blazen_uniffi_local_model_cgo_dispatchCallbackInterfaceForeignLocalModelMethod5),
	unloadAdapter: (C.UniffiCallbackInterfaceForeignLocalModelMethod6)(C.blazen_uniffi_local_model_cgo_dispatchCallbackInterfaceForeignLocalModelMethod6),
	listAdapters:  (C.UniffiCallbackInterfaceForeignLocalModelMethod7)(C.blazen_uniffi_local_model_cgo_dispatchCallbackInterfaceForeignLocalModelMethod7),
}

//export blazen_uniffi_local_model_cgo_dispatchCallbackInterfaceForeignLocalModelFree
func blazen_uniffi_local_model_cgo_dispatchCallbackInterfaceForeignLocalModelFree(handle C.uint64_t) {
	FfiConverterForeignLocalModelINSTANCE.handleMap.remove(uint64(handle))
}

//export blazen_uniffi_local_model_cgo_dispatchCallbackInterfaceForeignLocalModelClone
func blazen_uniffi_local_model_cgo_dispatchCallbackInterfaceForeignLocalModelClone(handle C.uint64_t) C.uint64_t {
	val, ok := FfiConverterForeignLocalModelINSTANCE.handleMap.tryGet(uint64(handle))
	if !ok {
		panic(fmt.Errorf("no callback in handle map: %d", handle))
	}
	return C.uint64_t(FfiConverterForeignLocalModelINSTANCE.handleMap.insert(val))
}

func (c FfiConverterForeignLocalModel) register() {
	C.uniffi_blazen_uniffi_fn_init_callback_vtable_foreignlocalmodel(&UniffiVTableCallbackInterfaceForeignLocalModelINSTANCE)
}

// Foreign-implementable training-progress sink.
//
// Modeled SYNC so the bridge to [`TrainingProgress`] (also sync) is
// trivial — the upstream trainer calls `on_event` from a tokio worker
// and an async foreign hop would require `block_on` from inside the
// same runtime (panic / deadlock-prone). Returning `Err(_)` cancels
// the run; the trainer surfaces it as `BlazenError::Cancelled`.
type ForeignTrainingProgress interface {
	OnEvent(event TrainingEventEnum) error
}

// Foreign-implementable training-progress sink.
//
// Modeled SYNC so the bridge to [`TrainingProgress`] (also sync) is
// trivial — the upstream trainer calls `on_event` from a tokio worker
// and an async foreign hop would require `block_on` from inside the
// same runtime (panic / deadlock-prone). Returning `Err(_)` cancels
// the run; the trainer surfaces it as `BlazenError::Cancelled`.
type ForeignTrainingProgressImpl struct {
	ffiObject FfiObject
}

func (_self *ForeignTrainingProgressImpl) OnEvent(event TrainingEventEnum) error {
	_pointer := _self.ffiObject.incrementPointer("ForeignTrainingProgress")
	defer _self.ffiObject.decrementPointer()
	_, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) bool {
		C.uniffi_blazen_uniffi_fn_method_foreigntrainingprogress_on_event(
			_pointer, FfiConverterTrainingEventEnumINSTANCE.Lower(event), _uniffiStatus)
		return false
	})
	return _uniffiErr.AsError()
}
func (object *ForeignTrainingProgressImpl) Destroy() {
	runtime.SetFinalizer(object, nil)
	object.ffiObject.destroy()
}

type FfiConverterForeignTrainingProgress struct {
	handleMap *concurrentHandleMap[ForeignTrainingProgress]
}

var FfiConverterForeignTrainingProgressINSTANCE = FfiConverterForeignTrainingProgress{
	handleMap: newConcurrentHandleMap[ForeignTrainingProgress](),
}

func (c FfiConverterForeignTrainingProgress) Lift(handle C.uint64_t) ForeignTrainingProgress {
	if uint64(handle)&1 == 0 {
		// Rust-generated handle (even), construct a new object wrapping the handle
		result := &ForeignTrainingProgressImpl{
			newFfiObject(
				handle,
				func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
					return C.uniffi_blazen_uniffi_fn_clone_foreigntrainingprogress(handle, status)
				},
				func(handle C.uint64_t, status *C.RustCallStatus) {
					C.uniffi_blazen_uniffi_fn_free_foreigntrainingprogress(handle, status)
				},
			),
		}
		runtime.SetFinalizer(result, (*ForeignTrainingProgressImpl).Destroy)
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

func (c FfiConverterForeignTrainingProgress) Read(reader io.Reader) ForeignTrainingProgress {
	return c.Lift(C.uint64_t(readUint64(reader)))
}

func (c FfiConverterForeignTrainingProgress) Lower(value ForeignTrainingProgress) C.uint64_t {
	// SAFETY (audited 2026-05-13): incrementPointer calls cloneFunction
	// which does Arc::clone on the Rust side, bumping the Rust refcount
	// independently of the Go-side callCounter. The defer below only
	// decrements the (redundant) Go counter; the returned handle survives
	// because the C caller owns its own Arc refcount via Arc::from_raw.
	if val, ok := value.(*ForeignTrainingProgressImpl); ok {
		// Rust-backed object, clone the handle
		handle := val.ffiObject.incrementPointer("ForeignTrainingProgress")
		defer val.ffiObject.decrementPointer()
		return handle
	} else {
		// Go-backed object, insert into handle map
		return C.uint64_t(c.handleMap.insert(value))
	}
}

func (c FfiConverterForeignTrainingProgress) Write(writer io.Writer, value ForeignTrainingProgress) {
	writeUint64(writer, uint64(c.Lower(value)))
}

func LiftFromExternalForeignTrainingProgress(handle uint64) ForeignTrainingProgress {
	return FfiConverterForeignTrainingProgressINSTANCE.Lift(C.uint64_t(handle))
}

func LowerToExternalForeignTrainingProgress(value ForeignTrainingProgress) uint64 {
	return uint64(FfiConverterForeignTrainingProgressINSTANCE.Lower(value))
}

type FfiDestroyerForeignTrainingProgress struct{}

func (_ FfiDestroyerForeignTrainingProgress) Destroy(value ForeignTrainingProgress) {
	if val, ok := value.(*ForeignTrainingProgressImpl); ok {
		val.Destroy()
	}
}

//export blazen_uniffi_manager_training_cgo_dispatchCallbackInterfaceForeignTrainingProgressMethod0
func blazen_uniffi_manager_training_cgo_dispatchCallbackInterfaceForeignTrainingProgressMethod0(uniffiHandle C.uint64_t, event C.RustBuffer, uniffiOutReturn *C.void, callStatus *C.RustCallStatus) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterForeignTrainingProgressINSTANCE.handleMap.tryGet(handle)
	if !ok {
		panic(fmt.Errorf("no callback in handle map: %d", handle))
	}

	err :=
		uniffiObj.OnEvent(
			FfiConverterTrainingEventEnumINSTANCE.Lift(GoRustBuffer{
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

}

var UniffiVTableCallbackInterfaceForeignTrainingProgressINSTANCE = C.UniffiVTableCallbackInterfaceForeignTrainingProgress{
	uniffiFree:  (C.UniffiCallbackInterfaceFree)(C.blazen_uniffi_manager_training_cgo_dispatchCallbackInterfaceForeignTrainingProgressFree),
	uniffiClone: (C.UniffiCallbackInterfaceClone)(C.blazen_uniffi_manager_training_cgo_dispatchCallbackInterfaceForeignTrainingProgressClone),
	onEvent:     (C.UniffiCallbackInterfaceForeignTrainingProgressMethod0)(C.blazen_uniffi_manager_training_cgo_dispatchCallbackInterfaceForeignTrainingProgressMethod0),
}

//export blazen_uniffi_manager_training_cgo_dispatchCallbackInterfaceForeignTrainingProgressFree
func blazen_uniffi_manager_training_cgo_dispatchCallbackInterfaceForeignTrainingProgressFree(handle C.uint64_t) {
	FfiConverterForeignTrainingProgressINSTANCE.handleMap.remove(uint64(handle))
}

//export blazen_uniffi_manager_training_cgo_dispatchCallbackInterfaceForeignTrainingProgressClone
func blazen_uniffi_manager_training_cgo_dispatchCallbackInterfaceForeignTrainingProgressClone(handle C.uint64_t) C.uint64_t {
	val, ok := FfiConverterForeignTrainingProgressINSTANCE.handleMap.tryGet(uint64(handle))
	if !ok {
		panic(fmt.Errorf("no callback in handle map: %d", handle))
	}
	return C.uint64_t(FfiConverterForeignTrainingProgressINSTANCE.handleMap.insert(val))
}

func (c FfiConverterForeignTrainingProgress) register() {
	C.uniffi_blazen_uniffi_fn_init_callback_vtable_foreigntrainingprogress(&UniffiVTableCallbackInterfaceForeignTrainingProgressINSTANCE)
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
	// SAFETY (audited 2026-05-13): incrementPointer calls cloneFunction
	// which does Arc::clone on the Rust side, bumping the Rust refcount
	// independently of the Go-side callCounter. The defer below only
	// decrements the (redundant) Go counter; the returned handle survives
	// because the C caller owns its own Arc refcount via Arc::from_raw.
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

// A chat completion model.
//
// Construct one via the per-provider factories in `providers.rs` (e.g.
// `Model::openai(options)` from the foreign-language side).
// Once obtained, call [`complete`](Self::complete) (async) or
// [`complete_blocking`](Self::complete_blocking) (sync) to generate
// responses.
type ModelInterface interface {
	// Perform a chat completion. Async on Swift / Kotlin; blocking on Go
	// (UniFFI's Go bindgen wraps the future in a goroutine-friendly call).
	Complete(request ModelRequest) (ModelResponse, error)
	// Synchronous variant of [`complete`](Self::complete) — blocks the
	// current thread on the shared Tokio runtime. Handy for Ruby scripts
	// and quick Go `main` functions where async machinery is overkill.
	// Prefer the async [`complete`](Self::complete) in long-running services.
	CompleteBlocking(request ModelRequest) (ModelResponse, error)
	// The model's identifier (e.g. `"gpt-4o"`, `"claude-3-5-sonnet"`).
	ModelId() string
}

// A chat completion model.
//
// Construct one via the per-provider factories in `providers.rs` (e.g.
// `Model::openai(options)` from the foreign-language side).
// Once obtained, call [`complete`](Self::complete) (async) or
// [`complete_blocking`](Self::complete_blocking) (sync) to generate
// responses.
type Model struct {
	ffiObject FfiObject
}

// Perform a chat completion. Async on Swift / Kotlin; blocking on Go
// (UniFFI's Go bindgen wraps the future in a goroutine-friendly call).
func (_self *Model) Complete(request ModelRequest) (ModelResponse, error) {
	_pointer := _self.ffiObject.incrementPointer("*Model")
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
		func(ffi RustBufferI) ModelResponse {
			return FfiConverterModelResponseINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_model_complete(
			_pointer, FfiConverterModelRequestINSTANCE.Lower(request)),
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
func (_self *Model) CompleteBlocking(request ModelRequest) (ModelResponse, error) {
	_pointer := _self.ffiObject.incrementPointer("*Model")
	defer _self.ffiObject.decrementPointer()
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) RustBufferI {
		return GoRustBuffer{
			inner: C.uniffi_blazen_uniffi_fn_method_model_complete_blocking(
				_pointer, FfiConverterModelRequestINSTANCE.Lower(request), _uniffiStatus),
		}
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue ModelResponse
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterModelResponseINSTANCE.Lift(_uniffiRV), nil
	}
}

// The model's identifier (e.g. `"gpt-4o"`, `"claude-3-5-sonnet"`).
func (_self *Model) ModelId() string {
	_pointer := _self.ffiObject.incrementPointer("*Model")
	defer _self.ffiObject.decrementPointer()
	return FfiConverterStringINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) RustBufferI {
		return GoRustBuffer{
			inner: C.uniffi_blazen_uniffi_fn_method_model_model_id(
				_pointer, _uniffiStatus),
		}
	}))
}
func (object *Model) Destroy() {
	runtime.SetFinalizer(object, nil)
	object.ffiObject.destroy()
}

type FfiConverterModel struct{}

var FfiConverterModelINSTANCE = FfiConverterModel{}

func (c FfiConverterModel) Lift(handle C.uint64_t) *Model {
	result := &Model{
		newFfiObject(
			handle,
			func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
				return C.uniffi_blazen_uniffi_fn_clone_model(handle, status)
			},
			func(handle C.uint64_t, status *C.RustCallStatus) {
				C.uniffi_blazen_uniffi_fn_free_model(handle, status)
			},
		),
	}
	runtime.SetFinalizer(result, (*Model).Destroy)
	return result
}

func (c FfiConverterModel) Read(reader io.Reader) *Model {
	return c.Lift(C.uint64_t(readUint64(reader)))
}

func (c FfiConverterModel) Lower(value *Model) C.uint64_t {
	// SAFETY (audited 2026-05-13): incrementPointer calls cloneFunction
	// which does Arc::clone on the Rust side, bumping the Rust refcount
	// independently of the Go-side callCounter. The defer below only
	// decrements the (redundant) Go counter; the returned handle survives
	// because the C caller owns its own Arc refcount via Arc::from_raw.
	handle := value.ffiObject.incrementPointer("*Model")
	defer value.ffiObject.decrementPointer()
	return handle
}

func (c FfiConverterModel) Write(writer io.Writer, value *Model) {
	writeUint64(writer, uint64(c.Lower(value)))
}

func LiftFromExternalModel(handle uint64) *Model {
	return FfiConverterModelINSTANCE.Lift(C.uint64_t(handle))
}

func LowerToExternalModel(value *Model) uint64 {
	return uint64(FfiConverterModelINSTANCE.Lower(value))
}

type FfiDestroyerModel struct{}

func (_ FfiDestroyerModel) Destroy(value *Model) {
	value.Destroy()
}

// gRPC client for the `BlazenModelServer` service, exposed to Go /
// Swift / Kotlin / Ruby via UniFFI.
//
// Wraps [`blazen_controlplane::ModelClient`] one-to-one; the upstream
// type is already cheaply cloneable and serialises concurrent RPCs
// internally, so the wrapper does not need its own mutex.
type ModelClientInterface interface {
	// Issue a non-streaming completion.
	//
	// `request_json` is the JSON form of
	// [`blazen_controlplane::model_protocol::CompleteRequest`]; the
	// `envelope_version` field is filled in automatically. Returns the
	// JSON form of
	// [`blazen_controlplane::model_protocol::CompleteResponse`].
	//
	// For streaming completions use a future wave's `stream_complete`
	// surface — this method always buffers the full response.
	//
	// # Errors
	// See [`Self::load_adapter`].
	Complete(requestJson string) (string, error)
	// Compute embeddings for one or more inputs.
	//
	// `request_json` is the JSON form of
	// [`blazen_controlplane::model_protocol::EmbedRequest`]; the
	// `envelope_version` field is filled in automatically. Returns the
	// JSON form of
	// [`blazen_controlplane::model_protocol::EmbedResponse`].
	//
	// # Errors
	// See [`Self::load_adapter`].
	Embed(requestJson string) (string, error)
	// Fetch a blob in one shot.
	//
	// `request_json` is the JSON form of
	// [`blazen_controlplane::model_protocol::FetchBlobRequest`]; the
	// `envelope_version` field is filled in automatically. The whole
	// response stream is buffered in memory: each
	// [`FetchBlobChunk::Data`] frame's bytes are concatenated and returned
	// as a single `Vec<u8>`; `Start` and `End` frames carry only metadata
	// and are not surfaced through this API.
	//
	// Callers that need to stream multi-gigabyte blobs without buffering
	// should drive [`blazen_controlplane::ModelClient::fetch_blob`]
	// directly from Rust.
	//
	// # Errors
	// Returns [`BlazenError::Validation`] when the request JSON cannot be
	// parsed; [`BlazenError::Peer`] for control-plane / transport failures
	// (either starting the stream or mid-stream).
	FetchBlob(requestJson string) ([]byte, error)
	// Generate one or more images.
	//
	// `request_json` is the JSON form of
	// [`blazen_controlplane::model_protocol::GenerateImageRequest`]; the
	// `envelope_version` field is filled in automatically. Returns the
	// JSON form of
	// [`blazen_controlplane::model_protocol::GenerateImageResponse`].
	//
	// # Errors
	// See [`Self::load_adapter`].
	GenerateImage(requestJson string) (string, error)
	// Generate music from a textual prompt.
	//
	// `request_json` is the JSON form of
	// [`blazen_controlplane::model_protocol::GenerateMusicRequest`]; the
	// `envelope_version` field is filled in automatically. Returns the
	// JSON form of
	// [`blazen_controlplane::model_protocol::GenerateMusicResponse`].
	//
	// # Errors
	// See [`Self::load_adapter`].
	GenerateMusic(requestJson string) (string, error)
	// Liveness check for a single model.
	//
	// # Errors
	// See [`Self::status`].
	IsLoaded(modelId string) (bool, error)
	// List adapters mounted on a model.
	//
	// `request_json` is the JSON form of
	// [`blazen_controlplane::model_protocol::ListAdaptersRequest`]; the
	// `envelope_version` field is filled in automatically. Returns the
	// JSON form of
	// [`blazen_controlplane::model_protocol::ListAdaptersResponse`].
	//
	// # Errors
	// See [`Self::load_adapter`].
	ListAdapters(requestJson string) (string, error)
	// Load a previously-registered model into its pool.
	//
	// # Errors
	// Returns [`BlazenError::Peer`] (`ControlPlaneTransport` /
	// `ControlPlaneRpc`) for wire or model-layer failures (e.g. unknown
	// `model_id`).
	Load(request LoadRecord) (LoadResultRecord, error)
	// Mount a LoRA / adapter onto a loaded model.
	//
	// `request_json` is the JSON form of
	// [`blazen_controlplane::model_protocol::LoadAdapterRequest`]; the
	// `envelope_version` field is filled in automatically and may be
	// omitted by the caller. Returns the JSON form of
	// [`blazen_controlplane::model_protocol::LoadAdapterResponse`].
	//
	// # Errors
	// Returns [`BlazenError::Validation`] when the request JSON cannot be
	// parsed or the response cannot be serialized;
	// [`BlazenError::Peer`] for control-plane / transport failures.
	LoadAdapter(requestJson string) (string, error)
	// Register-and-load a model directly from a Hugging Face Hub repo.
	// Returns the backend the loader chose (never
	// [`HfBackendHint::Auto`]).
	//
	// # Errors
	// See [`Self::load`]. Additionally surfaces loader-side failures
	// (HF fetch errors, unsupported repo layouts) via
	// `BlazenError::Peer` with `kind = "ControlPlaneRpc"`.
	LoadFromHf(request LoadFromHfRecord) (LoadResultRecord, error)
	// Fetch the server's view of registered models.
	//
	// When `model_id` is `Some(id)`, the response is filtered to just
	// that model (empty `models` vec if the id is unknown). When
	// `None`, every registered model is returned.
	//
	// # Errors
	// Returns [`BlazenError::Peer`] (`ControlPlaneTransport` /
	// `ControlPlaneRpc`) for wire or model-layer failures.
	Status(modelId *string) (StatusRecord, error)
	// Issue a streaming completion, delivering each token-delta to `sink`.
	//
	// `request_json` is the JSON form of
	// [`blazen_controlplane::model_protocol::CompleteRequest`]; the
	// `envelope_version` field is filled in automatically. As frames arrive
	// from the server, the [`StreamCompleteChunk::Delta`]'s `text` is
	// forwarded to [`CompletionStreamSink::on_chunk`] as the chunk's
	// `content_delta`; the terminal [`StreamCompleteChunk::Done`] triggers
	// [`CompletionStreamSink::on_done`] with the reported `finish_reason`
	// (empty string when the provider didn't supply one) and a
	// [`TokenUsage`] built from the `prompt_tokens` / `completion_tokens`
	// fields.
	//
	// Errors observed mid-stream are *delivered* via
	// [`CompletionStreamSink::on_error`] and the method returns `Ok(())`,
	// mirroring the symmetry of
	// [`crate::streaming::complete_streaming`]: the sink owns both
	// happy-path and error-path observation. The only way this method
	// itself returns `Err` is when the initial request JSON cannot be
	// parsed or the upstream `stream_complete` call fails to *start* the
	// stream.
	//
	// Reuses the existing text-only [`CompletionStreamSink`] so Go / Swift
	// / Kotlin / Ruby callers see a uniform streaming surface across both
	// the in-process [`crate::llm::Model`] path and the gRPC
	// [`ModelClient`] path. The wire-level
	// [`StreamCompleteChunk`] carries only `text` payloads today (no
	// per-frame tool-call deltas, citations, or reasoning trace), so the
	// text-only sink loses no information; if a future wire schema grows
	// structured fields, callers that need them should drive the gRPC
	// client directly.
	//
	// # Errors
	// Returns [`BlazenError::Validation`] when the request JSON cannot be
	// parsed; [`BlazenError::Peer`] for control-plane / transport failures
	// starting the stream.
	StreamComplete(requestJson string, sink CompletionStreamSink) error
	// Synthesize speech from text.
	//
	// `request_json` is the JSON form of
	// [`blazen_controlplane::model_protocol::TextToSpeechRequest`]; the
	// `envelope_version` field is filled in automatically. Returns the
	// JSON form of
	// [`blazen_controlplane::model_protocol::TextToSpeechResponse`].
	//
	// # Errors
	// See [`Self::load_adapter`].
	TextToSpeech(requestJson string) (string, error)
	// Transcribe audio to text.
	//
	// `request_json` is the JSON form of
	// [`blazen_controlplane::model_protocol::TranscribeRequest`]; the
	// `envelope_version` field is filled in automatically. Returns the
	// JSON form of
	// [`blazen_controlplane::model_protocol::TranscribeResponse`].
	//
	// # Errors
	// See [`Self::load_adapter`].
	Transcribe(requestJson string) (string, error)
	// Drop a previously-loaded model from memory.
	//
	// # Errors
	// See [`Self::load`].
	Unload(modelId string) error
	// Drop a previously-mounted adapter.
	//
	// `request_json` is the JSON form of
	// [`blazen_controlplane::model_protocol::UnloadAdapterRequest`]; the
	// `envelope_version` field is filled in automatically. Returns the
	// JSON form of
	// [`blazen_controlplane::model_protocol::UnloadAdapterResponse`].
	//
	// # Errors
	// See [`Self::load_adapter`].
	UnloadAdapter(requestJson string) (string, error)
	// Upload a blob in one shot.
	//
	// The entire `data` payload is buffered in memory and sent as a single
	// `UploadBlobChunk::Data` frame between a `Start` (carrying `blob_id`
	// + `mime`) and `End` frame. Returns the JSON form of
	// [`blazen_controlplane::model_protocol::UploadBlobResponse`] (the
	// server's ack, echoing the blob id + bytes received).
	//
	// This buffered surface is the simple path for the UniFFI bindings —
	// the whole payload must fit in process memory on both sides. Callers
	// pushing multi-gigabyte blobs (e.g. base model weights) should drive
	// [`blazen_controlplane::ModelClient::upload_blob`] directly from Rust
	// where they can construct the chunk stream incrementally.
	//
	// # Errors
	// Returns [`BlazenError::Peer`] for control-plane / transport
	// failures, or [`BlazenError::Validation`] when the response cannot be
	// serialized.
	UploadBlob(blobId string, mime string, data []byte) (string, error)
}

// gRPC client for the `BlazenModelServer` service, exposed to Go /
// Swift / Kotlin / Ruby via UniFFI.
//
// Wraps [`blazen_controlplane::ModelClient`] one-to-one; the upstream
// type is already cheaply cloneable and serialises concurrent RPCs
// internally, so the wrapper does not need its own mutex.
type ModelClient struct {
	ffiObject FfiObject
}

// Open a plaintext connection to a `BlazenModelServer` at
// `endpoint` (e.g. `"http://127.0.0.1:7070"`).
//
// # Errors
// Returns a [`BlazenError::Peer`] with `kind =
// "ControlPlaneTransport"` when the endpoint URI is invalid or the
// TCP / HTTP-2 handshake fails.
func ModelClientConnect(endpoint string) (*ModelClient, error) {
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
			res := C.ffi_blazen_uniffi_rust_future_complete_u64(handle, status)
			return res
		},
		// liftFn
		func(ffi C.uint64_t) *ModelClient {
			return FfiConverterModelClientINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_constructor_modelclient_connect(FfiConverterStringINSTANCE.Lower(endpoint)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_u64(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_u64(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// Open a TLS / mTLS connection. `ca_cert_pem` is the trust root
// the client uses to verify the server; `client_cert_pem` and
// `client_key_pem` are the client identity for mutual TLS (pass
// both or neither).
//
// # Errors
// Returns a [`BlazenError::Peer`] with `kind = "ControlPlaneTls"`
// when the PEM material can't be parsed, or with
// `kind = "ControlPlaneTransport"` for handshake failures.
// [`BlazenError::Validation`] if exactly one of `client_cert_pem`
// / `client_key_pem` is supplied.
func ModelClientConnectWithTls(endpoint string, caCertPem string, clientCertPem *string, clientKeyPem *string) (*ModelClient, error) {
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
			res := C.ffi_blazen_uniffi_rust_future_complete_u64(handle, status)
			return res
		},
		// liftFn
		func(ffi C.uint64_t) *ModelClient {
			return FfiConverterModelClientINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_constructor_modelclient_connect_with_tls(FfiConverterStringINSTANCE.Lower(endpoint), FfiConverterStringINSTANCE.Lower(caCertPem), FfiConverterOptionalStringINSTANCE.Lower(clientCertPem), FfiConverterOptionalStringINSTANCE.Lower(clientKeyPem)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_u64(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_u64(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// Issue a non-streaming completion.
//
// `request_json` is the JSON form of
// [`blazen_controlplane::model_protocol::CompleteRequest`]; the
// `envelope_version` field is filled in automatically. Returns the
// JSON form of
// [`blazen_controlplane::model_protocol::CompleteResponse`].
//
// For streaming completions use a future wave's `stream_complete`
// surface — this method always buffers the full response.
//
// # Errors
// See [`Self::load_adapter`].
func (_self *ModelClient) Complete(requestJson string) (string, error) {
	_pointer := _self.ffiObject.incrementPointer("*ModelClient")
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
		C.uniffi_blazen_uniffi_fn_method_modelclient_complete(
			_pointer, FfiConverterStringINSTANCE.Lower(requestJson)),
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

// Compute embeddings for one or more inputs.
//
// `request_json` is the JSON form of
// [`blazen_controlplane::model_protocol::EmbedRequest`]; the
// `envelope_version` field is filled in automatically. Returns the
// JSON form of
// [`blazen_controlplane::model_protocol::EmbedResponse`].
//
// # Errors
// See [`Self::load_adapter`].
func (_self *ModelClient) Embed(requestJson string) (string, error) {
	_pointer := _self.ffiObject.incrementPointer("*ModelClient")
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
		C.uniffi_blazen_uniffi_fn_method_modelclient_embed(
			_pointer, FfiConverterStringINSTANCE.Lower(requestJson)),
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

// Fetch a blob in one shot.
//
// `request_json` is the JSON form of
// [`blazen_controlplane::model_protocol::FetchBlobRequest`]; the
// `envelope_version` field is filled in automatically. The whole
// response stream is buffered in memory: each
// [`FetchBlobChunk::Data`] frame's bytes are concatenated and returned
// as a single `Vec<u8>`; `Start` and `End` frames carry only metadata
// and are not surfaced through this API.
//
// Callers that need to stream multi-gigabyte blobs without buffering
// should drive [`blazen_controlplane::ModelClient::fetch_blob`]
// directly from Rust.
//
// # Errors
// Returns [`BlazenError::Validation`] when the request JSON cannot be
// parsed; [`BlazenError::Peer`] for control-plane / transport failures
// (either starting the stream or mid-stream).
func (_self *ModelClient) FetchBlob(requestJson string) ([]byte, error) {
	_pointer := _self.ffiObject.incrementPointer("*ModelClient")
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
		func(ffi RustBufferI) []byte {
			return FfiConverterBytesINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_modelclient_fetch_blob(
			_pointer, FfiConverterStringINSTANCE.Lower(requestJson)),
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

// Generate one or more images.
//
// `request_json` is the JSON form of
// [`blazen_controlplane::model_protocol::GenerateImageRequest`]; the
// `envelope_version` field is filled in automatically. Returns the
// JSON form of
// [`blazen_controlplane::model_protocol::GenerateImageResponse`].
//
// # Errors
// See [`Self::load_adapter`].
func (_self *ModelClient) GenerateImage(requestJson string) (string, error) {
	_pointer := _self.ffiObject.incrementPointer("*ModelClient")
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
		C.uniffi_blazen_uniffi_fn_method_modelclient_generate_image(
			_pointer, FfiConverterStringINSTANCE.Lower(requestJson)),
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

// Generate music from a textual prompt.
//
// `request_json` is the JSON form of
// [`blazen_controlplane::model_protocol::GenerateMusicRequest`]; the
// `envelope_version` field is filled in automatically. Returns the
// JSON form of
// [`blazen_controlplane::model_protocol::GenerateMusicResponse`].
//
// # Errors
// See [`Self::load_adapter`].
func (_self *ModelClient) GenerateMusic(requestJson string) (string, error) {
	_pointer := _self.ffiObject.incrementPointer("*ModelClient")
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
		C.uniffi_blazen_uniffi_fn_method_modelclient_generate_music(
			_pointer, FfiConverterStringINSTANCE.Lower(requestJson)),
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

// Liveness check for a single model.
//
// # Errors
// See [`Self::status`].
func (_self *ModelClient) IsLoaded(modelId string) (bool, error) {
	_pointer := _self.ffiObject.incrementPointer("*ModelClient")
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
		C.uniffi_blazen_uniffi_fn_method_modelclient_is_loaded(
			_pointer, FfiConverterStringINSTANCE.Lower(modelId)),
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

// List adapters mounted on a model.
//
// `request_json` is the JSON form of
// [`blazen_controlplane::model_protocol::ListAdaptersRequest`]; the
// `envelope_version` field is filled in automatically. Returns the
// JSON form of
// [`blazen_controlplane::model_protocol::ListAdaptersResponse`].
//
// # Errors
// See [`Self::load_adapter`].
func (_self *ModelClient) ListAdapters(requestJson string) (string, error) {
	_pointer := _self.ffiObject.incrementPointer("*ModelClient")
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
		C.uniffi_blazen_uniffi_fn_method_modelclient_list_adapters(
			_pointer, FfiConverterStringINSTANCE.Lower(requestJson)),
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

// Load a previously-registered model into its pool.
//
// # Errors
// Returns [`BlazenError::Peer`] (`ControlPlaneTransport` /
// `ControlPlaneRpc`) for wire or model-layer failures (e.g. unknown
// `model_id`).
func (_self *ModelClient) Load(request LoadRecord) (LoadResultRecord, error) {
	_pointer := _self.ffiObject.incrementPointer("*ModelClient")
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
		func(ffi RustBufferI) LoadResultRecord {
			return FfiConverterLoadResultRecordINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_modelclient_load(
			_pointer, FfiConverterLoadRecordINSTANCE.Lower(request)),
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

// Mount a LoRA / adapter onto a loaded model.
//
// `request_json` is the JSON form of
// [`blazen_controlplane::model_protocol::LoadAdapterRequest`]; the
// `envelope_version` field is filled in automatically and may be
// omitted by the caller. Returns the JSON form of
// [`blazen_controlplane::model_protocol::LoadAdapterResponse`].
//
// # Errors
// Returns [`BlazenError::Validation`] when the request JSON cannot be
// parsed or the response cannot be serialized;
// [`BlazenError::Peer`] for control-plane / transport failures.
func (_self *ModelClient) LoadAdapter(requestJson string) (string, error) {
	_pointer := _self.ffiObject.incrementPointer("*ModelClient")
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
		C.uniffi_blazen_uniffi_fn_method_modelclient_load_adapter(
			_pointer, FfiConverterStringINSTANCE.Lower(requestJson)),
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

// Register-and-load a model directly from a Hugging Face Hub repo.
// Returns the backend the loader chose (never
// [`HfBackendHint::Auto`]).
//
// # Errors
// See [`Self::load`]. Additionally surfaces loader-side failures
// (HF fetch errors, unsupported repo layouts) via
// `BlazenError::Peer` with `kind = "ControlPlaneRpc"`.
func (_self *ModelClient) LoadFromHf(request LoadFromHfRecord) (LoadResultRecord, error) {
	_pointer := _self.ffiObject.incrementPointer("*ModelClient")
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
		func(ffi RustBufferI) LoadResultRecord {
			return FfiConverterLoadResultRecordINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_modelclient_load_from_hf(
			_pointer, FfiConverterLoadFromHfRecordINSTANCE.Lower(request)),
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

// Fetch the server's view of registered models.
//
// When `model_id` is `Some(id)`, the response is filtered to just
// that model (empty `models` vec if the id is unknown). When
// `None`, every registered model is returned.
//
// # Errors
// Returns [`BlazenError::Peer`] (`ControlPlaneTransport` /
// `ControlPlaneRpc`) for wire or model-layer failures.
func (_self *ModelClient) Status(modelId *string) (StatusRecord, error) {
	_pointer := _self.ffiObject.incrementPointer("*ModelClient")
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
		func(ffi RustBufferI) StatusRecord {
			return FfiConverterStatusRecordINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_modelclient_status(
			_pointer, FfiConverterOptionalStringINSTANCE.Lower(modelId)),
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

// Issue a streaming completion, delivering each token-delta to `sink`.
//
// `request_json` is the JSON form of
// [`blazen_controlplane::model_protocol::CompleteRequest`]; the
// `envelope_version` field is filled in automatically. As frames arrive
// from the server, the [`StreamCompleteChunk::Delta`]'s `text` is
// forwarded to [`CompletionStreamSink::on_chunk`] as the chunk's
// `content_delta`; the terminal [`StreamCompleteChunk::Done`] triggers
// [`CompletionStreamSink::on_done`] with the reported `finish_reason`
// (empty string when the provider didn't supply one) and a
// [`TokenUsage`] built from the `prompt_tokens` / `completion_tokens`
// fields.
//
// Errors observed mid-stream are *delivered* via
// [`CompletionStreamSink::on_error`] and the method returns `Ok(())`,
// mirroring the symmetry of
// [`crate::streaming::complete_streaming`]: the sink owns both
// happy-path and error-path observation. The only way this method
// itself returns `Err` is when the initial request JSON cannot be
// parsed or the upstream `stream_complete` call fails to *start* the
// stream.
//
// Reuses the existing text-only [`CompletionStreamSink`] so Go / Swift
// / Kotlin / Ruby callers see a uniform streaming surface across both
// the in-process [`crate::llm::Model`] path and the gRPC
// [`ModelClient`] path. The wire-level
// [`StreamCompleteChunk`] carries only `text` payloads today (no
// per-frame tool-call deltas, citations, or reasoning trace), so the
// text-only sink loses no information; if a future wire schema grows
// structured fields, callers that need them should drive the gRPC
// client directly.
//
// # Errors
// Returns [`BlazenError::Validation`] when the request JSON cannot be
// parsed; [`BlazenError::Peer`] for control-plane / transport failures
// starting the stream.
func (_self *ModelClient) StreamComplete(requestJson string, sink CompletionStreamSink) error {
	_pointer := _self.ffiObject.incrementPointer("*ModelClient")
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
		C.uniffi_blazen_uniffi_fn_method_modelclient_stream_complete(
			_pointer, FfiConverterStringINSTANCE.Lower(requestJson), FfiConverterCompletionStreamSinkINSTANCE.Lower(sink)),
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

// Synthesize speech from text.
//
// `request_json` is the JSON form of
// [`blazen_controlplane::model_protocol::TextToSpeechRequest`]; the
// `envelope_version` field is filled in automatically. Returns the
// JSON form of
// [`blazen_controlplane::model_protocol::TextToSpeechResponse`].
//
// # Errors
// See [`Self::load_adapter`].
func (_self *ModelClient) TextToSpeech(requestJson string) (string, error) {
	_pointer := _self.ffiObject.incrementPointer("*ModelClient")
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
		C.uniffi_blazen_uniffi_fn_method_modelclient_text_to_speech(
			_pointer, FfiConverterStringINSTANCE.Lower(requestJson)),
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
//
// `request_json` is the JSON form of
// [`blazen_controlplane::model_protocol::TranscribeRequest`]; the
// `envelope_version` field is filled in automatically. Returns the
// JSON form of
// [`blazen_controlplane::model_protocol::TranscribeResponse`].
//
// # Errors
// See [`Self::load_adapter`].
func (_self *ModelClient) Transcribe(requestJson string) (string, error) {
	_pointer := _self.ffiObject.incrementPointer("*ModelClient")
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
		C.uniffi_blazen_uniffi_fn_method_modelclient_transcribe(
			_pointer, FfiConverterStringINSTANCE.Lower(requestJson)),
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

// Drop a previously-loaded model from memory.
//
// # Errors
// See [`Self::load`].
func (_self *ModelClient) Unload(modelId string) error {
	_pointer := _self.ffiObject.incrementPointer("*ModelClient")
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
		C.uniffi_blazen_uniffi_fn_method_modelclient_unload(
			_pointer, FfiConverterStringINSTANCE.Lower(modelId)),
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

// Drop a previously-mounted adapter.
//
// `request_json` is the JSON form of
// [`blazen_controlplane::model_protocol::UnloadAdapterRequest`]; the
// `envelope_version` field is filled in automatically. Returns the
// JSON form of
// [`blazen_controlplane::model_protocol::UnloadAdapterResponse`].
//
// # Errors
// See [`Self::load_adapter`].
func (_self *ModelClient) UnloadAdapter(requestJson string) (string, error) {
	_pointer := _self.ffiObject.incrementPointer("*ModelClient")
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
		C.uniffi_blazen_uniffi_fn_method_modelclient_unload_adapter(
			_pointer, FfiConverterStringINSTANCE.Lower(requestJson)),
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

// Upload a blob in one shot.
//
// The entire `data` payload is buffered in memory and sent as a single
// `UploadBlobChunk::Data` frame between a `Start` (carrying `blob_id`
// + `mime`) and `End` frame. Returns the JSON form of
// [`blazen_controlplane::model_protocol::UploadBlobResponse`] (the
// server's ack, echoing the blob id + bytes received).
//
// This buffered surface is the simple path for the UniFFI bindings —
// the whole payload must fit in process memory on both sides. Callers
// pushing multi-gigabyte blobs (e.g. base model weights) should drive
// [`blazen_controlplane::ModelClient::upload_blob`] directly from Rust
// where they can construct the chunk stream incrementally.
//
// # Errors
// Returns [`BlazenError::Peer`] for control-plane / transport
// failures, or [`BlazenError::Validation`] when the response cannot be
// serialized.
func (_self *ModelClient) UploadBlob(blobId string, mime string, data []byte) (string, error) {
	_pointer := _self.ffiObject.incrementPointer("*ModelClient")
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
		C.uniffi_blazen_uniffi_fn_method_modelclient_upload_blob(
			_pointer, FfiConverterStringINSTANCE.Lower(blobId), FfiConverterStringINSTANCE.Lower(mime), FfiConverterBytesINSTANCE.Lower(data)),
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
func (object *ModelClient) Destroy() {
	runtime.SetFinalizer(object, nil)
	object.ffiObject.destroy()
}

type FfiConverterModelClient struct{}

var FfiConverterModelClientINSTANCE = FfiConverterModelClient{}

func (c FfiConverterModelClient) Lift(handle C.uint64_t) *ModelClient {
	result := &ModelClient{
		newFfiObject(
			handle,
			func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
				return C.uniffi_blazen_uniffi_fn_clone_modelclient(handle, status)
			},
			func(handle C.uint64_t, status *C.RustCallStatus) {
				C.uniffi_blazen_uniffi_fn_free_modelclient(handle, status)
			},
		),
	}
	runtime.SetFinalizer(result, (*ModelClient).Destroy)
	return result
}

func (c FfiConverterModelClient) Read(reader io.Reader) *ModelClient {
	return c.Lift(C.uint64_t(readUint64(reader)))
}

func (c FfiConverterModelClient) Lower(value *ModelClient) C.uint64_t {
	// SAFETY (audited 2026-05-13): incrementPointer calls cloneFunction
	// which does Arc::clone on the Rust side, bumping the Rust refcount
	// independently of the Go-side callCounter. The defer below only
	// decrements the (redundant) Go counter; the returned handle survives
	// because the C caller owns its own Arc refcount via Arc::from_raw.
	handle := value.ffiObject.incrementPointer("*ModelClient")
	defer value.ffiObject.decrementPointer()
	return handle
}

func (c FfiConverterModelClient) Write(writer io.Writer, value *ModelClient) {
	writeUint64(writer, uint64(c.Lower(value)))
}

func LiftFromExternalModelClient(handle uint64) *ModelClient {
	return FfiConverterModelClientINSTANCE.Lift(C.uint64_t(handle))
}

func LowerToExternalModelClient(value *ModelClient) uint64 {
	return uint64(FfiConverterModelClientINSTANCE.Lower(value))
}

type FfiDestroyerModelClient struct{}

func (_ FfiDestroyerModelClient) Destroy(value *ModelClient) {
	value.Destroy()
}

// A music / sound-effect generation model.
//
// Construct via one of the per-backend factory functions
// ([`new_musicgen_model`], [`new_stable_audio_model`],
// [`new_audiogen_model`], or [`new_fal_music_model`]). Use the async
// [`generate_music`](Self::generate_music) / [`generate_sfx`](Self::generate_sfx)
// methods for one-shot rendering, or [`stream_generate_music_to_sink`] /
// [`stream_generate_sfx_to_sink`] for chunk-level streaming.
type MusicModelInterface interface {
	// Generate `duration_seconds` of music conditioned on `prompt`.
	GenerateMusic(prompt string, durationSeconds float32) (MusicResult, error)
	// Synchronous variant of [`generate_music`](Self::generate_music).
	GenerateMusicBlocking(prompt string, durationSeconds float32) (MusicResult, error)
	// Generate `duration_seconds` of sound-effect audio conditioned on
	// `prompt`.
	GenerateSfx(prompt string, durationSeconds float32) (MusicResult, error)
	// Synchronous variant of [`generate_sfx`](Self::generate_sfx).
	GenerateSfxBlocking(prompt string, durationSeconds float32) (MusicResult, error)
}

// A music / sound-effect generation model.
//
// Construct via one of the per-backend factory functions
// ([`new_musicgen_model`], [`new_stable_audio_model`],
// [`new_audiogen_model`], or [`new_fal_music_model`]). Use the async
// [`generate_music`](Self::generate_music) / [`generate_sfx`](Self::generate_sfx)
// methods for one-shot rendering, or [`stream_generate_music_to_sink`] /
// [`stream_generate_sfx_to_sink`] for chunk-level streaming.
type MusicModel struct {
	ffiObject FfiObject
}

// Generate `duration_seconds` of music conditioned on `prompt`.
func (_self *MusicModel) GenerateMusic(prompt string, durationSeconds float32) (MusicResult, error) {
	_pointer := _self.ffiObject.incrementPointer("*MusicModel")
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
		func(ffi RustBufferI) MusicResult {
			return FfiConverterMusicResultINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_musicmodel_generate_music(
			_pointer, FfiConverterStringINSTANCE.Lower(prompt), FfiConverterFloat32INSTANCE.Lower(durationSeconds)),
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

// Synchronous variant of [`generate_music`](Self::generate_music).
func (_self *MusicModel) GenerateMusicBlocking(prompt string, durationSeconds float32) (MusicResult, error) {
	_pointer := _self.ffiObject.incrementPointer("*MusicModel")
	defer _self.ffiObject.decrementPointer()
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) RustBufferI {
		return GoRustBuffer{
			inner: C.uniffi_blazen_uniffi_fn_method_musicmodel_generate_music_blocking(
				_pointer, FfiConverterStringINSTANCE.Lower(prompt), FfiConverterFloat32INSTANCE.Lower(durationSeconds), _uniffiStatus),
		}
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue MusicResult
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterMusicResultINSTANCE.Lift(_uniffiRV), nil
	}
}

// Generate `duration_seconds` of sound-effect audio conditioned on
// `prompt`.
func (_self *MusicModel) GenerateSfx(prompt string, durationSeconds float32) (MusicResult, error) {
	_pointer := _self.ffiObject.incrementPointer("*MusicModel")
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
		func(ffi RustBufferI) MusicResult {
			return FfiConverterMusicResultINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_musicmodel_generate_sfx(
			_pointer, FfiConverterStringINSTANCE.Lower(prompt), FfiConverterFloat32INSTANCE.Lower(durationSeconds)),
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

// Synchronous variant of [`generate_sfx`](Self::generate_sfx).
func (_self *MusicModel) GenerateSfxBlocking(prompt string, durationSeconds float32) (MusicResult, error) {
	_pointer := _self.ffiObject.incrementPointer("*MusicModel")
	defer _self.ffiObject.decrementPointer()
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) RustBufferI {
		return GoRustBuffer{
			inner: C.uniffi_blazen_uniffi_fn_method_musicmodel_generate_sfx_blocking(
				_pointer, FfiConverterStringINSTANCE.Lower(prompt), FfiConverterFloat32INSTANCE.Lower(durationSeconds), _uniffiStatus),
		}
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue MusicResult
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterMusicResultINSTANCE.Lift(_uniffiRV), nil
	}
}
func (object *MusicModel) Destroy() {
	runtime.SetFinalizer(object, nil)
	object.ffiObject.destroy()
}

type FfiConverterMusicModel struct{}

var FfiConverterMusicModelINSTANCE = FfiConverterMusicModel{}

func (c FfiConverterMusicModel) Lift(handle C.uint64_t) *MusicModel {
	result := &MusicModel{
		newFfiObject(
			handle,
			func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
				return C.uniffi_blazen_uniffi_fn_clone_musicmodel(handle, status)
			},
			func(handle C.uint64_t, status *C.RustCallStatus) {
				C.uniffi_blazen_uniffi_fn_free_musicmodel(handle, status)
			},
		),
	}
	runtime.SetFinalizer(result, (*MusicModel).Destroy)
	return result
}

func (c FfiConverterMusicModel) Read(reader io.Reader) *MusicModel {
	return c.Lift(C.uint64_t(readUint64(reader)))
}

func (c FfiConverterMusicModel) Lower(value *MusicModel) C.uint64_t {
	// SAFETY (audited 2026-05-13): incrementPointer calls cloneFunction
	// which does Arc::clone on the Rust side, bumping the Rust refcount
	// independently of the Go-side callCounter. The defer below only
	// decrements the (redundant) Go counter; the returned handle survives
	// because the C caller owns its own Arc refcount via Arc::from_raw.
	handle := value.ffiObject.incrementPointer("*MusicModel")
	defer value.ffiObject.decrementPointer()
	return handle
}

func (c FfiConverterMusicModel) Write(writer io.Writer, value *MusicModel) {
	writeUint64(writer, uint64(c.Lower(value)))
}

func LiftFromExternalMusicModel(handle uint64) *MusicModel {
	return FfiConverterMusicModelINSTANCE.Lift(C.uint64_t(handle))
}

func LowerToExternalMusicModel(value *MusicModel) uint64 {
	return uint64(FfiConverterMusicModelINSTANCE.Lower(value))
}

type FfiDestroyerMusicModel struct{}

func (_ FfiDestroyerMusicModel) Destroy(value *MusicModel) {
	value.Destroy()
}

// Sink for streaming music / SFX output, implemented in foreign code.
//
// Symmetric to [`crate::streaming::CompletionStreamSink`]: the streaming
// engine calls [`on_chunk`](Self::on_chunk) for each emitted chunk, then
// exactly one of [`on_done`](Self::on_done) or [`on_error`](Self::on_error).
// Implementations should treat the terminal callbacks as cleanup hooks
// (close channels, complete async iterators, signal flow completion, ...).
type MusicStreamSink interface {
	// Receive a single chunk from the streaming response.
	//
	// Returning an `Err` aborts the stream — the engine delivers the error
	// via [`on_error`](Self::on_error) and stops dispatching further
	// chunks.
	OnChunk(chunk MusicChunk) error
	// Receive the terminal completion signal. Called exactly once at the
	// end of a successful stream.
	OnDone() error
	// Receive a fatal error from the stream. Called exactly once when the
	// stream fails midway (or fails to start at all).
	OnError(cause *BlazenError) error
}

// Sink for streaming music / SFX output, implemented in foreign code.
//
// Symmetric to [`crate::streaming::CompletionStreamSink`]: the streaming
// engine calls [`on_chunk`](Self::on_chunk) for each emitted chunk, then
// exactly one of [`on_done`](Self::on_done) or [`on_error`](Self::on_error).
// Implementations should treat the terminal callbacks as cleanup hooks
// (close channels, complete async iterators, signal flow completion, ...).
type MusicStreamSinkImpl struct {
	ffiObject FfiObject
}

// Receive a single chunk from the streaming response.
//
// Returning an `Err` aborts the stream — the engine delivers the error
// via [`on_error`](Self::on_error) and stops dispatching further
// chunks.
func (_self *MusicStreamSinkImpl) OnChunk(chunk MusicChunk) error {
	_pointer := _self.ffiObject.incrementPointer("MusicStreamSink")
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
		C.uniffi_blazen_uniffi_fn_method_musicstreamsink_on_chunk(
			_pointer, FfiConverterMusicChunkINSTANCE.Lower(chunk)),
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
// end of a successful stream.
func (_self *MusicStreamSinkImpl) OnDone() error {
	_pointer := _self.ffiObject.incrementPointer("MusicStreamSink")
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
		C.uniffi_blazen_uniffi_fn_method_musicstreamsink_on_done(
			_pointer),
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
// stream fails midway (or fails to start at all).
func (_self *MusicStreamSinkImpl) OnError(cause *BlazenError) error {
	_pointer := _self.ffiObject.incrementPointer("MusicStreamSink")
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
		C.uniffi_blazen_uniffi_fn_method_musicstreamsink_on_error(
			_pointer, FfiConverterBlazenErrorINSTANCE.Lower(cause)),
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
func (object *MusicStreamSinkImpl) Destroy() {
	runtime.SetFinalizer(object, nil)
	object.ffiObject.destroy()
}

type FfiConverterMusicStreamSink struct {
	handleMap *concurrentHandleMap[MusicStreamSink]
}

var FfiConverterMusicStreamSinkINSTANCE = FfiConverterMusicStreamSink{
	handleMap: newConcurrentHandleMap[MusicStreamSink](),
}

func (c FfiConverterMusicStreamSink) Lift(handle C.uint64_t) MusicStreamSink {
	if uint64(handle)&1 == 0 {
		// Rust-generated handle (even), construct a new object wrapping the handle
		result := &MusicStreamSinkImpl{
			newFfiObject(
				handle,
				func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
					return C.uniffi_blazen_uniffi_fn_clone_musicstreamsink(handle, status)
				},
				func(handle C.uint64_t, status *C.RustCallStatus) {
					C.uniffi_blazen_uniffi_fn_free_musicstreamsink(handle, status)
				},
			),
		}
		runtime.SetFinalizer(result, (*MusicStreamSinkImpl).Destroy)
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

func (c FfiConverterMusicStreamSink) Read(reader io.Reader) MusicStreamSink {
	return c.Lift(C.uint64_t(readUint64(reader)))
}

func (c FfiConverterMusicStreamSink) Lower(value MusicStreamSink) C.uint64_t {
	// SAFETY (audited 2026-05-13): incrementPointer calls cloneFunction
	// which does Arc::clone on the Rust side, bumping the Rust refcount
	// independently of the Go-side callCounter. The defer below only
	// decrements the (redundant) Go counter; the returned handle survives
	// because the C caller owns its own Arc refcount via Arc::from_raw.
	if val, ok := value.(*MusicStreamSinkImpl); ok {
		// Rust-backed object, clone the handle
		handle := val.ffiObject.incrementPointer("MusicStreamSink")
		defer val.ffiObject.decrementPointer()
		return handle
	} else {
		// Go-backed object, insert into handle map
		return C.uint64_t(c.handleMap.insert(value))
	}
}

func (c FfiConverterMusicStreamSink) Write(writer io.Writer, value MusicStreamSink) {
	writeUint64(writer, uint64(c.Lower(value)))
}

func LiftFromExternalMusicStreamSink(handle uint64) MusicStreamSink {
	return FfiConverterMusicStreamSinkINSTANCE.Lift(C.uint64_t(handle))
}

func LowerToExternalMusicStreamSink(value MusicStreamSink) uint64 {
	return uint64(FfiConverterMusicStreamSinkINSTANCE.Lower(value))
}

type FfiDestroyerMusicStreamSink struct{}

func (_ FfiDestroyerMusicStreamSink) Destroy(value MusicStreamSink) {
	if val, ok := value.(*MusicStreamSinkImpl); ok {
		val.Destroy()
	}
}

//export blazen_uniffi_compute_music_cgo_dispatchCallbackInterfaceMusicStreamSinkMethod0
func blazen_uniffi_compute_music_cgo_dispatchCallbackInterfaceMusicStreamSinkMethod0(uniffiHandle C.uint64_t, chunk C.RustBuffer, uniffiFutureCallback C.UniffiForeignFutureCompleteVoid, uniffiCallbackData C.uint64_t, uniffiOutDroppedCallback *C.UniffiForeignFutureDroppedCallbackStruct) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterMusicStreamSinkINSTANCE.handleMap.tryGet(handle)
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
				FfiConverterMusicChunkINSTANCE.Lift(GoRustBuffer{
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

//export blazen_uniffi_compute_music_cgo_dispatchCallbackInterfaceMusicStreamSinkMethod1
func blazen_uniffi_compute_music_cgo_dispatchCallbackInterfaceMusicStreamSinkMethod1(uniffiHandle C.uint64_t, uniffiFutureCallback C.UniffiForeignFutureCompleteVoid, uniffiCallbackData C.uint64_t, uniffiOutDroppedCallback *C.UniffiForeignFutureDroppedCallbackStruct) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterMusicStreamSinkINSTANCE.handleMap.tryGet(handle)
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
			uniffiObj.OnDone()

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

//export blazen_uniffi_compute_music_cgo_dispatchCallbackInterfaceMusicStreamSinkMethod2
func blazen_uniffi_compute_music_cgo_dispatchCallbackInterfaceMusicStreamSinkMethod2(uniffiHandle C.uint64_t, cause C.RustBuffer, uniffiFutureCallback C.UniffiForeignFutureCompleteVoid, uniffiCallbackData C.uint64_t, uniffiOutDroppedCallback *C.UniffiForeignFutureDroppedCallbackStruct) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterMusicStreamSinkINSTANCE.handleMap.tryGet(handle)
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
					inner: cause,
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

var UniffiVTableCallbackInterfaceMusicStreamSinkINSTANCE = C.UniffiVTableCallbackInterfaceMusicStreamSink{
	uniffiFree:  (C.UniffiCallbackInterfaceFree)(C.blazen_uniffi_compute_music_cgo_dispatchCallbackInterfaceMusicStreamSinkFree),
	uniffiClone: (C.UniffiCallbackInterfaceClone)(C.blazen_uniffi_compute_music_cgo_dispatchCallbackInterfaceMusicStreamSinkClone),
	onChunk:     (C.UniffiCallbackInterfaceMusicStreamSinkMethod0)(C.blazen_uniffi_compute_music_cgo_dispatchCallbackInterfaceMusicStreamSinkMethod0),
	onDone:      (C.UniffiCallbackInterfaceMusicStreamSinkMethod1)(C.blazen_uniffi_compute_music_cgo_dispatchCallbackInterfaceMusicStreamSinkMethod1),
	onError:     (C.UniffiCallbackInterfaceMusicStreamSinkMethod2)(C.blazen_uniffi_compute_music_cgo_dispatchCallbackInterfaceMusicStreamSinkMethod2),
}

//export blazen_uniffi_compute_music_cgo_dispatchCallbackInterfaceMusicStreamSinkFree
func blazen_uniffi_compute_music_cgo_dispatchCallbackInterfaceMusicStreamSinkFree(handle C.uint64_t) {
	FfiConverterMusicStreamSinkINSTANCE.handleMap.remove(uint64(handle))
}

//export blazen_uniffi_compute_music_cgo_dispatchCallbackInterfaceMusicStreamSinkClone
func blazen_uniffi_compute_music_cgo_dispatchCallbackInterfaceMusicStreamSinkClone(handle C.uint64_t) C.uint64_t {
	val, ok := FfiConverterMusicStreamSinkINSTANCE.handleMap.tryGet(uint64(handle))
	if !ok {
		panic(fmt.Errorf("no callback in handle map: %d", handle))
	}
	return C.uint64_t(FfiConverterMusicStreamSinkINSTANCE.handleMap.insert(val))
}

func (c FfiConverterMusicStreamSink) register() {
	C.uniffi_blazen_uniffi_fn_init_callback_vtable_musicstreamsink(&UniffiVTableCallbackInterfaceMusicStreamSinkINSTANCE)
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
	// SAFETY (audited 2026-05-13): incrementPointer calls cloneFunction
	// which does Arc::clone on the Rust side, bumping the Rust refcount
	// independently of the Go-side callCounter. The defer below only
	// decrements the (redundant) Go counter; the returned handle survives
	// because the C caller owns its own Arc refcount via Arc::from_raw.
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
	// SAFETY (audited 2026-05-13): incrementPointer calls cloneFunction
	// which does Arc::clone on the Rust side, bumping the Rust refcount
	// independently of the Go-side callCounter. The defer below only
	// decrements the (redundant) Go counter; the returned handle survives
	// because the C caller owns its own Arc refcount via Arc::from_raw.
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
	// SAFETY (audited 2026-05-13): incrementPointer calls cloneFunction
	// which does Arc::clone on the Rust side, bumping the Rust refcount
	// independently of the Go-side callCounter. The defer below only
	// decrements the (redundant) Go counter; the returned handle survives
	// because the C caller owns its own Arc refcount via Arc::from_raw.
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
	// SAFETY (audited 2026-05-13): incrementPointer calls cloneFunction
	// which does Arc::clone on the Rust side, bumping the Rust refcount
	// independently of the Go-side callCounter. The defer below only
	// decrements the (redundant) Go counter; the returned handle survives
	// because the C caller owns its own Arc refcount via Arc::from_raw.
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
	// SAFETY (audited 2026-05-13): incrementPointer calls cloneFunction
	// which does Arc::clone on the Rust side, bumping the Rust refcount
	// independently of the Go-side callCounter. The defer below only
	// decrements the (redundant) Go counter; the returned handle survives
	// because the C caller owns its own Arc refcount via Arc::from_raw.
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
	// SAFETY (audited 2026-05-13): incrementPointer calls cloneFunction
	// which does Arc::clone on the Rust side, bumping the Rust refcount
	// independently of the Go-side callCounter. The defer below only
	// decrements the (redundant) Go counter; the returned handle survives
	// because the C caller owns its own Arc refcount via Arc::from_raw.
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
	//
	// ## Structured `ToolOutput`
	//
	// Returning a JSON object with a `data` key opts into the structured
	// [`blazen_llm::types::ToolOutput`] shape:
	//
	// ```text
	// {
	// "data": { /* user-visible payload */ },
	// "llm_override": {
	// "kind": "parts",
	// "parts": [{ "type": "text", "text": "summary for the model" }]
	// }
	// }
	// ```
	//
	// `llmOverride` (camelCase) is also accepted. The inner `parts[]`
	// discriminator uses `"type"` (matching the core `ContentPart` serde
	// tag); the outer discriminator uses `"kind"`. Foreign-language helper
	// types (`Blazen::ToolOutput` in Ruby, `blazen.ToolOutput` in Go,
	// `Blazen.ToolOutput` in Swift, `dev.blazen.ToolOutput` in Kotlin)
	// produce this shape automatically.
	//
	// Returning anything else (a primitive, a JSON object without a `data`
	// key, etc.) auto-wraps the whole value as `data` with no override.
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
//
// ## Structured `ToolOutput`
//
// Returning a JSON object with a `data` key opts into the structured
// [`blazen_llm::types::ToolOutput`] shape:
//
// ```text
// {
// "data": { /* user-visible payload */ },
// "llm_override": {
// "kind": "parts",
// "parts": [{ "type": "text", "text": "summary for the model" }]
// }
// }
// ```
//
// `llmOverride` (camelCase) is also accepted. The inner `parts[]`
// discriminator uses `"type"` (matching the core `ContentPart` serde
// tag); the outer discriminator uses `"kind"`. Foreign-language helper
// types (`Blazen::ToolOutput` in Ruby, `blazen.ToolOutput` in Go,
// `Blazen.ToolOutput` in Swift, `dev.blazen.ToolOutput` in Kotlin)
// produce this shape automatically.
//
// Returning anything else (a primitive, a JSON object without a `data`
// key, etc.) auto-wraps the whole value as `data` with no override.
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
	// SAFETY (audited 2026-05-13): incrementPointer calls cloneFunction
	// which does Arc::clone on the Rust side, bumping the Rust refcount
	// independently of the Go-side callCounter. The defer below only
	// decrements the (redundant) Go counter; the returned handle survives
	// because the C caller owns its own Arc refcount via Arc::from_raw.
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
	// SAFETY (audited 2026-05-13): incrementPointer calls cloneFunction
	// which does Arc::clone on the Rust side, bumping the Rust refcount
	// independently of the Go-side callCounter. The defer below only
	// decrements the (redundant) Go counter; the returned handle survives
	// because the C caller owns its own Arc refcount via Arc::from_raw.
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

// JSONL-backed training dataset opaque handle.
//
// Construct via [`UniffiJsonlDataset::from_path`] and pass to
// [`UniffiModelManager::train_lora`]. The dataset is reference-counted
// (`Arc`-shared), so foreign callers can keep a handle around and
// re-use it across multiple training runs.
type UniffiJsonlDatasetInterface interface {
	IsEmpty() bool
	// Number of examples in the dataset.
	Len() uint64
}

// JSONL-backed training dataset opaque handle.
//
// Construct via [`UniffiJsonlDataset::from_path`] and pass to
// [`UniffiModelManager::train_lora`]. The dataset is reference-counted
// (`Arc`-shared), so foreign callers can keep a handle around and
// re-use it across multiple training runs.
type UniffiJsonlDataset struct {
	ffiObject FfiObject
}

// Load a JSONL training file using the tokenizer at `tokenizer_path`.
//
// `chat_template` is optional Jinja2 from `tokenizer_config.json`;
// required if any row uses the OpenAI `messages` shape.
// `device` matches the trainer device strings — `"cpu"`,
// `"cuda"` / `"cuda:N"`, `"metal"` / `"metal:N"` (default `"cpu"`).
func UniffiJsonlDatasetFromPath(path string, tokenizerPath string, chatTemplate *string, maxSeqLen uint32, device *string, padTokenId uint32) (*UniffiJsonlDataset, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_constructor_uniffijsonldataset_from_path(FfiConverterStringINSTANCE.Lower(path), FfiConverterStringINSTANCE.Lower(tokenizerPath), FfiConverterOptionalStringINSTANCE.Lower(chatTemplate), FfiConverterUint32INSTANCE.Lower(maxSeqLen), FfiConverterOptionalStringINSTANCE.Lower(device), FfiConverterUint32INSTANCE.Lower(padTokenId), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *UniffiJsonlDataset
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterUniffiJsonlDatasetINSTANCE.Lift(_uniffiRV), nil
	}
}

func (_self *UniffiJsonlDataset) IsEmpty() bool {
	_pointer := _self.ffiObject.incrementPointer("*UniffiJsonlDataset")
	defer _self.ffiObject.decrementPointer()
	return FfiConverterBoolINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) C.int8_t {
		return C.uniffi_blazen_uniffi_fn_method_uniffijsonldataset_is_empty(
			_pointer, _uniffiStatus)
	}))
}

// Number of examples in the dataset.
func (_self *UniffiJsonlDataset) Len() uint64 {
	_pointer := _self.ffiObject.incrementPointer("*UniffiJsonlDataset")
	defer _self.ffiObject.decrementPointer()
	return FfiConverterUint64INSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_method_uniffijsonldataset_len(
			_pointer, _uniffiStatus)
	}))
}
func (object *UniffiJsonlDataset) Destroy() {
	runtime.SetFinalizer(object, nil)
	object.ffiObject.destroy()
}

type FfiConverterUniffiJsonlDataset struct{}

var FfiConverterUniffiJsonlDatasetINSTANCE = FfiConverterUniffiJsonlDataset{}

func (c FfiConverterUniffiJsonlDataset) Lift(handle C.uint64_t) *UniffiJsonlDataset {
	result := &UniffiJsonlDataset{
		newFfiObject(
			handle,
			func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
				return C.uniffi_blazen_uniffi_fn_clone_uniffijsonldataset(handle, status)
			},
			func(handle C.uint64_t, status *C.RustCallStatus) {
				C.uniffi_blazen_uniffi_fn_free_uniffijsonldataset(handle, status)
			},
		),
	}
	runtime.SetFinalizer(result, (*UniffiJsonlDataset).Destroy)
	return result
}

func (c FfiConverterUniffiJsonlDataset) Read(reader io.Reader) *UniffiJsonlDataset {
	return c.Lift(C.uint64_t(readUint64(reader)))
}

func (c FfiConverterUniffiJsonlDataset) Lower(value *UniffiJsonlDataset) C.uint64_t {
	// SAFETY (audited 2026-05-13): incrementPointer calls cloneFunction
	// which does Arc::clone on the Rust side, bumping the Rust refcount
	// independently of the Go-side callCounter. The defer below only
	// decrements the (redundant) Go counter; the returned handle survives
	// because the C caller owns its own Arc refcount via Arc::from_raw.
	handle := value.ffiObject.incrementPointer("*UniffiJsonlDataset")
	defer value.ffiObject.decrementPointer()
	return handle
}

func (c FfiConverterUniffiJsonlDataset) Write(writer io.Writer, value *UniffiJsonlDataset) {
	writeUint64(writer, uint64(c.Lower(value)))
}

func LiftFromExternalUniffiJsonlDataset(handle uint64) *UniffiJsonlDataset {
	return FfiConverterUniffiJsonlDatasetINSTANCE.Lift(C.uint64_t(handle))
}

func LowerToExternalUniffiJsonlDataset(value *UniffiJsonlDataset) uint64 {
	return uint64(FfiConverterUniffiJsonlDatasetINSTANCE.Lower(value))
}

type FfiDestroyerUniffiJsonlDataset struct{}

func (_ FfiDestroyerUniffiJsonlDataset) Destroy(value *UniffiJsonlDataset) {
	value.Destroy()
}

// Memory-budget-aware model manager with per-pool LRU eviction.
//
// Foreign code constructs one of these, registers
// [`ForeignLocalModel`]-implementing handles against it, and drives loads /
// unloads / adapter lifecycle from any thread / fiber / goroutine /
// coroutine on the foreign side.
type UniffiModelManagerInterface interface {
	AvailableBytes(pool string) (uint64, error)
	EnsureLoaded(modelId string) error
	IsLoaded(modelId string) bool
	ListAdapters(modelId string) ([]AdapterStatusRecord, error)
	Load(modelId string) error
	// Mount a PEFT-format LoRA adapter and return the adapter id reported
	// by the backend.
	LoadAdapter(modelId string, adapterDir string, options AdapterOptionsRecord) (string, error)
	// Synchronous variant of [`Self::load`] — blocks the current thread on
	// the shared Tokio runtime.
	LoadBlocking(modelId string) error
	// Probe a Hugging Face repo, pick a local-inference backend, build the
	// provider, and register it under `id`.
	//
	// Returns the chosen backend as a lower-case stable string
	// (`"mistralrs"` / `"candle"` / `"llamacpp"`). The model starts unloaded
	// — call [`Self::load`] or [`Self::ensure_loaded`] to materialize it.
	//
	// Errors on empty repo id, gated/missing repo, PEFT-adapter-only repo
	// (use [`Self::load_adapter`] instead), missing backend feature, or any
	// provider construction failure.
	LoadFromHf(id string, repo string, options HfLoadOptionsRecord) (string, error)
	// List configured pools and their budgets in bytes.
	Pools() []PoolStatusRecord
	// Register a foreign-implemented [`ForeignLocalModel`] under `id`.
	//
	// `memory_estimate_bytes` is the model's estimated footprint and is
	// charged against the pool derived from the foreign model's `device()`
	// when it's loaded.
	RegisterLocal(id string, model ForeignLocalModel, memoryEstimateBytes uint64) error
	Status() []ModelStatusRecord
	Unload(modelId string) error
	UnloadAdapter(modelId string, adapterId string) error
	// Synchronous variant of [`Self::unload`].
	UnloadBlocking(modelId string) error
	UsedBytes(pool string) (uint64, error)
	// Run a full fine-tune (every parameter trainable; no `LoRA`
	// adapter).
	//
	// Returns [`FullFineTuneResultRecord`] rather than
	// [`TrainedAdapterRecord`] because the output is a complete set
	// of model weights in `config.core.output_dir`, not a mountable
	// PEFT delta. Setting `config.gradient_checkpointing = true` is
	// rejected up-front because candle 0.10.2 has no activation-
	// checkpointing primitive.
	FineTune(config FullFineTuneConfigRecord, dataset *UniffiJsonlDataset, progress *ForeignTrainingProgress) (FullFineTuneResultRecord, error)
	// Train a `LoRA` adapter via Direct Preference Optimization (DPO).
	//
	// Downloads the base model from `HuggingFace` (cached) plus the
	// reference model (defaults to `config.core.base_model_repo`),
	// runs the DPO training loop driven by `dataset`, and writes the
	// resulting PEFT-format adapter to `config.core.output_dir`.
	//
	// If `progress` is provided, its `on_event` is called for each
	// transition. Returning `Err(_)` from the callback cancels the run
	// with [`BlazenError::Cancelled`].
	TrainDpo(config DpoConfigRecord, dataset *UniffiPreferenceJsonlDataset, progress *ForeignTrainingProgress) (TrainedAdapterRecord, error)
	// Train a `LoRA` adapter via Kahneman-Tversky Optimization (KTO).
	//
	// Like DPO, KTO requires a frozen reference model; the dataset
	// schema differs: each row is a `(prompt, completion, desirable)`
	// triple.
	TrainKto(config KtoConfigRecord, dataset *UniffiRatedJsonlDataset, progress *ForeignTrainingProgress) (TrainedAdapterRecord, error)
	// Train a LoRA adapter end-to-end on the configured base model.
	//
	// Downloads the base model from HuggingFace (cached), runs the
	// AdamW + LoRA training loop driven by `dataset`, and writes the
	// resulting PEFT-format adapter to `config.output_dir`. The
	// returned [`TrainedAdapterRecord`] points at an on-disk adapter
	// directory that's immediately mountable via
	// [`UniffiModelManager::load_adapter`] on a compatible backend.
	//
	// If `progress` is provided, its `on_event` is called for each
	// Started / StepCompleted / Evaluating / EvalCompleted /
	// CheckpointSaved / Finished transition. Returning `Err(_)` from
	// the callback cancels the run with [`BlazenError::Cancelled`].
	TrainLora(config TrainConfigRecord, dataset *UniffiJsonlDataset, progress *ForeignTrainingProgress) (TrainedAdapterRecord, error)
	// Train a `LoRA` adapter via Odds Ratio Preference Optimization
	// (ORPO). Reference-free — combines an SFT loss on chosen
	// completions with an odds-ratio preference term.
	TrainOrpo(config OrpoConfigRecord, dataset *UniffiPreferenceJsonlDataset, progress *ForeignTrainingProgress) (TrainedAdapterRecord, error)
	// Train a `LoRA` adapter via Simple Preference Optimization
	// (`SimPO`). Reference-free and length-normalized.
	TrainSimpo(config SimpoConfigRecord, dataset *UniffiPreferenceJsonlDataset, progress *ForeignTrainingProgress) (TrainedAdapterRecord, error)
}

// Memory-budget-aware model manager with per-pool LRU eviction.
//
// Foreign code constructs one of these, registers
// [`ForeignLocalModel`]-implementing handles against it, and drives loads /
// unloads / adapter lifecycle from any thread / fiber / goroutine /
// coroutine on the foreign side.
type UniffiModelManager struct {
	ffiObject FfiObject
}

// Construct a manager with no budget enforcement (both `Cpu` and
// `Gpu(0)` pools seeded with `u64::MAX`).
//
// Matches the Python binding's `ModelManager()` no-arg sentinel
// behaviour. For real deployments, prefer
// [`Self::with_budgets_gb`](Self::with_budgets_gb) or
// [`Self::with_pool_budgets`](Self::with_pool_budgets).
func NewUniffiModelManager() *UniffiModelManager {
	return FfiConverterUniffiModelManagerINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_constructor_uniffimodelmanager_new(_uniffiStatus)
	}))
}

// Construct a manager with one CPU-pool budget and one GPU-pool
// (`Gpu(0)`) budget, both expressed in gigabytes.
func UniffiModelManagerWithBudgetsGb(cpuRamGb float64, gpuVramGb float64) *UniffiModelManager {
	return FfiConverterUniffiModelManagerINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_constructor_uniffimodelmanager_with_budgets_gb(FfiConverterFloat64INSTANCE.Lower(cpuRamGb), FfiConverterFloat64INSTANCE.Lower(gpuVramGb), _uniffiStatus)
	}))
}

// Construct a manager with explicit per-pool budgets.
//
// Keys are pool labels (`"cpu"`, `"gpu"`, `"gpu:0"`, `"gpu:1"`, ...);
// values are budgets in **gigabytes** (mirrors the Python binding's
// `pool_budgets` ergonomics — bytes-as-`u64` would force foreign
// callers to write `64 * 1024 * 1024 * 1024` for trivial values).
func UniffiModelManagerWithPoolBudgets(perPoolBudgets map[string]float64) (*UniffiModelManager, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_constructor_uniffimodelmanager_with_pool_budgets(FfiConverterMapStringFloat64INSTANCE.Lower(perPoolBudgets), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *UniffiModelManager
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterUniffiModelManagerINSTANCE.Lift(_uniffiRV), nil
	}
}

func (_self *UniffiModelManager) AvailableBytes(pool string) (uint64, error) {
	_pointer := _self.ffiObject.incrementPointer("*UniffiModelManager")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
			res := C.ffi_blazen_uniffi_rust_future_complete_u64(handle, status)
			return res
		},
		// liftFn
		func(ffi C.uint64_t) uint64 {
			return FfiConverterUint64INSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_uniffimodelmanager_available_bytes(
			_pointer, FfiConverterStringINSTANCE.Lower(pool)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_u64(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_u64(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

func (_self *UniffiModelManager) EnsureLoaded(modelId string) error {
	_pointer := _self.ffiObject.incrementPointer("*UniffiModelManager")
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
		C.uniffi_blazen_uniffi_fn_method_uniffimodelmanager_ensure_loaded(
			_pointer, FfiConverterStringINSTANCE.Lower(modelId)),
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

func (_self *UniffiModelManager) IsLoaded(modelId string) bool {
	_pointer := _self.ffiObject.incrementPointer("*UniffiModelManager")
	defer _self.ffiObject.decrementPointer()
	res, _ := uniffiRustCallAsync[error](
		nil,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) C.int8_t {
			res := C.ffi_blazen_uniffi_rust_future_complete_i8(handle, status)
			return res
		},
		// liftFn
		func(ffi C.int8_t) bool {
			return FfiConverterBoolINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_uniffimodelmanager_is_loaded(
			_pointer, FfiConverterStringINSTANCE.Lower(modelId)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_i8(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_i8(handle)
		},
	)

	return res
}

func (_self *UniffiModelManager) ListAdapters(modelId string) ([]AdapterStatusRecord, error) {
	_pointer := _self.ffiObject.incrementPointer("*UniffiModelManager")
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
		func(ffi RustBufferI) []AdapterStatusRecord {
			return FfiConverterSequenceAdapterStatusRecordINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_uniffimodelmanager_list_adapters(
			_pointer, FfiConverterStringINSTANCE.Lower(modelId)),
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

func (_self *UniffiModelManager) Load(modelId string) error {
	_pointer := _self.ffiObject.incrementPointer("*UniffiModelManager")
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
		C.uniffi_blazen_uniffi_fn_method_uniffimodelmanager_load(
			_pointer, FfiConverterStringINSTANCE.Lower(modelId)),
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

// Mount a PEFT-format LoRA adapter and return the adapter id reported
// by the backend.
func (_self *UniffiModelManager) LoadAdapter(modelId string, adapterDir string, options AdapterOptionsRecord) (string, error) {
	_pointer := _self.ffiObject.incrementPointer("*UniffiModelManager")
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
		C.uniffi_blazen_uniffi_fn_method_uniffimodelmanager_load_adapter(
			_pointer, FfiConverterStringINSTANCE.Lower(modelId), FfiConverterStringINSTANCE.Lower(adapterDir), FfiConverterAdapterOptionsRecordINSTANCE.Lower(options)),
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

// Synchronous variant of [`Self::load`] — blocks the current thread on
// the shared Tokio runtime.
func (_self *UniffiModelManager) LoadBlocking(modelId string) error {
	_pointer := _self.ffiObject.incrementPointer("*UniffiModelManager")
	defer _self.ffiObject.decrementPointer()
	_, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) bool {
		C.uniffi_blazen_uniffi_fn_method_uniffimodelmanager_load_blocking(
			_pointer, FfiConverterStringINSTANCE.Lower(modelId), _uniffiStatus)
		return false
	})
	return _uniffiErr.AsError()
}

// Probe a Hugging Face repo, pick a local-inference backend, build the
// provider, and register it under `id`.
//
// Returns the chosen backend as a lower-case stable string
// (`"mistralrs"` / `"candle"` / `"llamacpp"`). The model starts unloaded
// — call [`Self::load`] or [`Self::ensure_loaded`] to materialize it.
//
// Errors on empty repo id, gated/missing repo, PEFT-adapter-only repo
// (use [`Self::load_adapter`] instead), missing backend feature, or any
// provider construction failure.
func (_self *UniffiModelManager) LoadFromHf(id string, repo string, options HfLoadOptionsRecord) (string, error) {
	_pointer := _self.ffiObject.incrementPointer("*UniffiModelManager")
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
		C.uniffi_blazen_uniffi_fn_method_uniffimodelmanager_load_from_hf(
			_pointer, FfiConverterStringINSTANCE.Lower(id), FfiConverterStringINSTANCE.Lower(repo), FfiConverterHfLoadOptionsRecordINSTANCE.Lower(options)),
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

// List configured pools and their budgets in bytes.
func (_self *UniffiModelManager) Pools() []PoolStatusRecord {
	_pointer := _self.ffiObject.incrementPointer("*UniffiModelManager")
	defer _self.ffiObject.decrementPointer()
	return FfiConverterSequencePoolStatusRecordINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) RustBufferI {
		return GoRustBuffer{
			inner: C.uniffi_blazen_uniffi_fn_method_uniffimodelmanager_pools(
				_pointer, _uniffiStatus),
		}
	}))
}

// Register a foreign-implemented [`ForeignLocalModel`] under `id`.
//
// `memory_estimate_bytes` is the model's estimated footprint and is
// charged against the pool derived from the foreign model's `device()`
// when it's loaded.
func (_self *UniffiModelManager) RegisterLocal(id string, model ForeignLocalModel, memoryEstimateBytes uint64) error {
	_pointer := _self.ffiObject.incrementPointer("*UniffiModelManager")
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
		C.uniffi_blazen_uniffi_fn_method_uniffimodelmanager_register_local(
			_pointer, FfiConverterStringINSTANCE.Lower(id), FfiConverterForeignLocalModelINSTANCE.Lower(model), FfiConverterUint64INSTANCE.Lower(memoryEstimateBytes)),
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

func (_self *UniffiModelManager) Status() []ModelStatusRecord {
	_pointer := _self.ffiObject.incrementPointer("*UniffiModelManager")
	defer _self.ffiObject.decrementPointer()
	res, _ := uniffiRustCallAsync[error](
		nil,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) RustBufferI {
			res := C.ffi_blazen_uniffi_rust_future_complete_rust_buffer(handle, status)
			return GoRustBuffer{
				inner: res,
			}
		},
		// liftFn
		func(ffi RustBufferI) []ModelStatusRecord {
			return FfiConverterSequenceModelStatusRecordINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_uniffimodelmanager_status(
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

	return res
}

func (_self *UniffiModelManager) Unload(modelId string) error {
	_pointer := _self.ffiObject.incrementPointer("*UniffiModelManager")
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
		C.uniffi_blazen_uniffi_fn_method_uniffimodelmanager_unload(
			_pointer, FfiConverterStringINSTANCE.Lower(modelId)),
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

func (_self *UniffiModelManager) UnloadAdapter(modelId string, adapterId string) error {
	_pointer := _self.ffiObject.incrementPointer("*UniffiModelManager")
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
		C.uniffi_blazen_uniffi_fn_method_uniffimodelmanager_unload_adapter(
			_pointer, FfiConverterStringINSTANCE.Lower(modelId), FfiConverterStringINSTANCE.Lower(adapterId)),
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

// Synchronous variant of [`Self::unload`].
func (_self *UniffiModelManager) UnloadBlocking(modelId string) error {
	_pointer := _self.ffiObject.incrementPointer("*UniffiModelManager")
	defer _self.ffiObject.decrementPointer()
	_, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) bool {
		C.uniffi_blazen_uniffi_fn_method_uniffimodelmanager_unload_blocking(
			_pointer, FfiConverterStringINSTANCE.Lower(modelId), _uniffiStatus)
		return false
	})
	return _uniffiErr.AsError()
}

func (_self *UniffiModelManager) UsedBytes(pool string) (uint64, error) {
	_pointer := _self.ffiObject.incrementPointer("*UniffiModelManager")
	defer _self.ffiObject.decrementPointer()
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
			res := C.ffi_blazen_uniffi_rust_future_complete_u64(handle, status)
			return res
		},
		// liftFn
		func(ffi C.uint64_t) uint64 {
			return FfiConverterUint64INSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_uniffimodelmanager_used_bytes(
			_pointer, FfiConverterStringINSTANCE.Lower(pool)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_u64(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_u64(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
}

// Run a full fine-tune (every parameter trainable; no `LoRA`
// adapter).
//
// Returns [`FullFineTuneResultRecord`] rather than
// [`TrainedAdapterRecord`] because the output is a complete set
// of model weights in `config.core.output_dir`, not a mountable
// PEFT delta. Setting `config.gradient_checkpointing = true` is
// rejected up-front because candle 0.10.2 has no activation-
// checkpointing primitive.
func (_self *UniffiModelManager) FineTune(config FullFineTuneConfigRecord, dataset *UniffiJsonlDataset, progress *ForeignTrainingProgress) (FullFineTuneResultRecord, error) {
	_pointer := _self.ffiObject.incrementPointer("*UniffiModelManager")
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
		func(ffi RustBufferI) FullFineTuneResultRecord {
			return FfiConverterFullFineTuneResultRecordINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_uniffimodelmanager_fine_tune(
			_pointer, FfiConverterFullFineTuneConfigRecordINSTANCE.Lower(config), FfiConverterUniffiJsonlDatasetINSTANCE.Lower(dataset), FfiConverterOptionalForeignTrainingProgressINSTANCE.Lower(progress)),
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

// Train a `LoRA` adapter via Direct Preference Optimization (DPO).
//
// Downloads the base model from `HuggingFace` (cached) plus the
// reference model (defaults to `config.core.base_model_repo`),
// runs the DPO training loop driven by `dataset`, and writes the
// resulting PEFT-format adapter to `config.core.output_dir`.
//
// If `progress` is provided, its `on_event` is called for each
// transition. Returning `Err(_)` from the callback cancels the run
// with [`BlazenError::Cancelled`].
func (_self *UniffiModelManager) TrainDpo(config DpoConfigRecord, dataset *UniffiPreferenceJsonlDataset, progress *ForeignTrainingProgress) (TrainedAdapterRecord, error) {
	_pointer := _self.ffiObject.incrementPointer("*UniffiModelManager")
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
		func(ffi RustBufferI) TrainedAdapterRecord {
			return FfiConverterTrainedAdapterRecordINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_uniffimodelmanager_train_dpo(
			_pointer, FfiConverterDpoConfigRecordINSTANCE.Lower(config), FfiConverterUniffiPreferenceJsonlDatasetINSTANCE.Lower(dataset), FfiConverterOptionalForeignTrainingProgressINSTANCE.Lower(progress)),
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

// Train a `LoRA` adapter via Kahneman-Tversky Optimization (KTO).
//
// Like DPO, KTO requires a frozen reference model; the dataset
// schema differs: each row is a `(prompt, completion, desirable)`
// triple.
func (_self *UniffiModelManager) TrainKto(config KtoConfigRecord, dataset *UniffiRatedJsonlDataset, progress *ForeignTrainingProgress) (TrainedAdapterRecord, error) {
	_pointer := _self.ffiObject.incrementPointer("*UniffiModelManager")
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
		func(ffi RustBufferI) TrainedAdapterRecord {
			return FfiConverterTrainedAdapterRecordINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_uniffimodelmanager_train_kto(
			_pointer, FfiConverterKtoConfigRecordINSTANCE.Lower(config), FfiConverterUniffiRatedJsonlDatasetINSTANCE.Lower(dataset), FfiConverterOptionalForeignTrainingProgressINSTANCE.Lower(progress)),
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

// Train a LoRA adapter end-to-end on the configured base model.
//
// Downloads the base model from HuggingFace (cached), runs the
// AdamW + LoRA training loop driven by `dataset`, and writes the
// resulting PEFT-format adapter to `config.output_dir`. The
// returned [`TrainedAdapterRecord`] points at an on-disk adapter
// directory that's immediately mountable via
// [`UniffiModelManager::load_adapter`] on a compatible backend.
//
// If `progress` is provided, its `on_event` is called for each
// Started / StepCompleted / Evaluating / EvalCompleted /
// CheckpointSaved / Finished transition. Returning `Err(_)` from
// the callback cancels the run with [`BlazenError::Cancelled`].
func (_self *UniffiModelManager) TrainLora(config TrainConfigRecord, dataset *UniffiJsonlDataset, progress *ForeignTrainingProgress) (TrainedAdapterRecord, error) {
	_pointer := _self.ffiObject.incrementPointer("*UniffiModelManager")
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
		func(ffi RustBufferI) TrainedAdapterRecord {
			return FfiConverterTrainedAdapterRecordINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_uniffimodelmanager_train_lora(
			_pointer, FfiConverterTrainConfigRecordINSTANCE.Lower(config), FfiConverterUniffiJsonlDatasetINSTANCE.Lower(dataset), FfiConverterOptionalForeignTrainingProgressINSTANCE.Lower(progress)),
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

// Train a `LoRA` adapter via Odds Ratio Preference Optimization
// (ORPO). Reference-free — combines an SFT loss on chosen
// completions with an odds-ratio preference term.
func (_self *UniffiModelManager) TrainOrpo(config OrpoConfigRecord, dataset *UniffiPreferenceJsonlDataset, progress *ForeignTrainingProgress) (TrainedAdapterRecord, error) {
	_pointer := _self.ffiObject.incrementPointer("*UniffiModelManager")
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
		func(ffi RustBufferI) TrainedAdapterRecord {
			return FfiConverterTrainedAdapterRecordINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_uniffimodelmanager_train_orpo(
			_pointer, FfiConverterOrpoConfigRecordINSTANCE.Lower(config), FfiConverterUniffiPreferenceJsonlDatasetINSTANCE.Lower(dataset), FfiConverterOptionalForeignTrainingProgressINSTANCE.Lower(progress)),
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

// Train a `LoRA` adapter via Simple Preference Optimization
// (`SimPO`). Reference-free and length-normalized.
func (_self *UniffiModelManager) TrainSimpo(config SimpoConfigRecord, dataset *UniffiPreferenceJsonlDataset, progress *ForeignTrainingProgress) (TrainedAdapterRecord, error) {
	_pointer := _self.ffiObject.incrementPointer("*UniffiModelManager")
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
		func(ffi RustBufferI) TrainedAdapterRecord {
			return FfiConverterTrainedAdapterRecordINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_uniffimodelmanager_train_simpo(
			_pointer, FfiConverterSimpoConfigRecordINSTANCE.Lower(config), FfiConverterUniffiPreferenceJsonlDatasetINSTANCE.Lower(dataset), FfiConverterOptionalForeignTrainingProgressINSTANCE.Lower(progress)),
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
func (object *UniffiModelManager) Destroy() {
	runtime.SetFinalizer(object, nil)
	object.ffiObject.destroy()
}

type FfiConverterUniffiModelManager struct{}

var FfiConverterUniffiModelManagerINSTANCE = FfiConverterUniffiModelManager{}

func (c FfiConverterUniffiModelManager) Lift(handle C.uint64_t) *UniffiModelManager {
	result := &UniffiModelManager{
		newFfiObject(
			handle,
			func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
				return C.uniffi_blazen_uniffi_fn_clone_uniffimodelmanager(handle, status)
			},
			func(handle C.uint64_t, status *C.RustCallStatus) {
				C.uniffi_blazen_uniffi_fn_free_uniffimodelmanager(handle, status)
			},
		),
	}
	runtime.SetFinalizer(result, (*UniffiModelManager).Destroy)
	return result
}

func (c FfiConverterUniffiModelManager) Read(reader io.Reader) *UniffiModelManager {
	return c.Lift(C.uint64_t(readUint64(reader)))
}

func (c FfiConverterUniffiModelManager) Lower(value *UniffiModelManager) C.uint64_t {
	// SAFETY (audited 2026-05-13): incrementPointer calls cloneFunction
	// which does Arc::clone on the Rust side, bumping the Rust refcount
	// independently of the Go-side callCounter. The defer below only
	// decrements the (redundant) Go counter; the returned handle survives
	// because the C caller owns its own Arc refcount via Arc::from_raw.
	handle := value.ffiObject.incrementPointer("*UniffiModelManager")
	defer value.ffiObject.decrementPointer()
	return handle
}

func (c FfiConverterUniffiModelManager) Write(writer io.Writer, value *UniffiModelManager) {
	writeUint64(writer, uint64(c.Lower(value)))
}

func LiftFromExternalUniffiModelManager(handle uint64) *UniffiModelManager {
	return FfiConverterUniffiModelManagerINSTANCE.Lift(C.uint64_t(handle))
}

func LowerToExternalUniffiModelManager(value *UniffiModelManager) uint64 {
	return uint64(FfiConverterUniffiModelManagerINSTANCE.Lower(value))
}

type FfiDestroyerUniffiModelManager struct{}

func (_ FfiDestroyerUniffiModelManager) Destroy(value *UniffiModelManager) {
	value.Destroy()
}

// JSONL-backed preference-pair dataset opaque handle for DPO / ORPO /
// `SimPO`.
//
// Each line of the input file must deserialize to either
// `{"prompt": "...", "chosen": "...", "rejected": "..."}` or
// `{"messages": [...], "chosen": "...", "rejected": "..."}` (the
// latter requires `chat_template`).
type UniffiPreferenceJsonlDatasetInterface interface {
	IsEmpty() bool
	// Number of preference examples in the dataset.
	Len() uint64
}

// JSONL-backed preference-pair dataset opaque handle for DPO / ORPO /
// `SimPO`.
//
// Each line of the input file must deserialize to either
// `{"prompt": "...", "chosen": "...", "rejected": "..."}` or
// `{"messages": [...], "chosen": "...", "rejected": "..."}` (the
// latter requires `chat_template`).
type UniffiPreferenceJsonlDataset struct {
	ffiObject FfiObject
}

// Load a preference-pair JSONL file using the tokenizer at
// `tokenizer_path`. Args mirror [`UniffiJsonlDataset::from_path`].
func UniffiPreferenceJsonlDatasetFromPath(path string, tokenizerPath string, chatTemplate *string, maxSeqLen uint32, device *string, padTokenId uint32) (*UniffiPreferenceJsonlDataset, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_constructor_uniffipreferencejsonldataset_from_path(FfiConverterStringINSTANCE.Lower(path), FfiConverterStringINSTANCE.Lower(tokenizerPath), FfiConverterOptionalStringINSTANCE.Lower(chatTemplate), FfiConverterUint32INSTANCE.Lower(maxSeqLen), FfiConverterOptionalStringINSTANCE.Lower(device), FfiConverterUint32INSTANCE.Lower(padTokenId), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *UniffiPreferenceJsonlDataset
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterUniffiPreferenceJsonlDatasetINSTANCE.Lift(_uniffiRV), nil
	}
}

func (_self *UniffiPreferenceJsonlDataset) IsEmpty() bool {
	_pointer := _self.ffiObject.incrementPointer("*UniffiPreferenceJsonlDataset")
	defer _self.ffiObject.decrementPointer()
	return FfiConverterBoolINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) C.int8_t {
		return C.uniffi_blazen_uniffi_fn_method_uniffipreferencejsonldataset_is_empty(
			_pointer, _uniffiStatus)
	}))
}

// Number of preference examples in the dataset.
func (_self *UniffiPreferenceJsonlDataset) Len() uint64 {
	_pointer := _self.ffiObject.incrementPointer("*UniffiPreferenceJsonlDataset")
	defer _self.ffiObject.decrementPointer()
	return FfiConverterUint64INSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_method_uniffipreferencejsonldataset_len(
			_pointer, _uniffiStatus)
	}))
}
func (object *UniffiPreferenceJsonlDataset) Destroy() {
	runtime.SetFinalizer(object, nil)
	object.ffiObject.destroy()
}

type FfiConverterUniffiPreferenceJsonlDataset struct{}

var FfiConverterUniffiPreferenceJsonlDatasetINSTANCE = FfiConverterUniffiPreferenceJsonlDataset{}

func (c FfiConverterUniffiPreferenceJsonlDataset) Lift(handle C.uint64_t) *UniffiPreferenceJsonlDataset {
	result := &UniffiPreferenceJsonlDataset{
		newFfiObject(
			handle,
			func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
				return C.uniffi_blazen_uniffi_fn_clone_uniffipreferencejsonldataset(handle, status)
			},
			func(handle C.uint64_t, status *C.RustCallStatus) {
				C.uniffi_blazen_uniffi_fn_free_uniffipreferencejsonldataset(handle, status)
			},
		),
	}
	runtime.SetFinalizer(result, (*UniffiPreferenceJsonlDataset).Destroy)
	return result
}

func (c FfiConverterUniffiPreferenceJsonlDataset) Read(reader io.Reader) *UniffiPreferenceJsonlDataset {
	return c.Lift(C.uint64_t(readUint64(reader)))
}

func (c FfiConverterUniffiPreferenceJsonlDataset) Lower(value *UniffiPreferenceJsonlDataset) C.uint64_t {
	// SAFETY (audited 2026-05-13): incrementPointer calls cloneFunction
	// which does Arc::clone on the Rust side, bumping the Rust refcount
	// independently of the Go-side callCounter. The defer below only
	// decrements the (redundant) Go counter; the returned handle survives
	// because the C caller owns its own Arc refcount via Arc::from_raw.
	handle := value.ffiObject.incrementPointer("*UniffiPreferenceJsonlDataset")
	defer value.ffiObject.decrementPointer()
	return handle
}

func (c FfiConverterUniffiPreferenceJsonlDataset) Write(writer io.Writer, value *UniffiPreferenceJsonlDataset) {
	writeUint64(writer, uint64(c.Lower(value)))
}

func LiftFromExternalUniffiPreferenceJsonlDataset(handle uint64) *UniffiPreferenceJsonlDataset {
	return FfiConverterUniffiPreferenceJsonlDatasetINSTANCE.Lift(C.uint64_t(handle))
}

func LowerToExternalUniffiPreferenceJsonlDataset(value *UniffiPreferenceJsonlDataset) uint64 {
	return uint64(FfiConverterUniffiPreferenceJsonlDatasetINSTANCE.Lower(value))
}

type FfiDestroyerUniffiPreferenceJsonlDataset struct{}

func (_ FfiDestroyerUniffiPreferenceJsonlDataset) Destroy(value *UniffiPreferenceJsonlDataset) {
	value.Destroy()
}

// JSONL-backed rated single-completion dataset opaque handle for KTO.
//
// Each line of the input file must deserialize to either
// `{"prompt": "...", "completion": "...", "label": true|false}` or
// `{"messages": [...], "completion": "...", "label": ...}` (the
// latter requires `chat_template`).
type UniffiRatedJsonlDatasetInterface interface {
	IsEmpty() bool
	// Number of rated examples in the dataset.
	Len() uint64
}

// JSONL-backed rated single-completion dataset opaque handle for KTO.
//
// Each line of the input file must deserialize to either
// `{"prompt": "...", "completion": "...", "label": true|false}` or
// `{"messages": [...], "completion": "...", "label": ...}` (the
// latter requires `chat_template`).
type UniffiRatedJsonlDataset struct {
	ffiObject FfiObject
}

// Load a rated JSONL file using the tokenizer at `tokenizer_path`.
// Args mirror [`UniffiJsonlDataset::from_path`].
func UniffiRatedJsonlDatasetFromPath(path string, tokenizerPath string, chatTemplate *string, maxSeqLen uint32, device *string, padTokenId uint32) (*UniffiRatedJsonlDataset, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_constructor_uniffiratedjsonldataset_from_path(FfiConverterStringINSTANCE.Lower(path), FfiConverterStringINSTANCE.Lower(tokenizerPath), FfiConverterOptionalStringINSTANCE.Lower(chatTemplate), FfiConverterUint32INSTANCE.Lower(maxSeqLen), FfiConverterOptionalStringINSTANCE.Lower(device), FfiConverterUint32INSTANCE.Lower(padTokenId), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *UniffiRatedJsonlDataset
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterUniffiRatedJsonlDatasetINSTANCE.Lift(_uniffiRV), nil
	}
}

func (_self *UniffiRatedJsonlDataset) IsEmpty() bool {
	_pointer := _self.ffiObject.incrementPointer("*UniffiRatedJsonlDataset")
	defer _self.ffiObject.decrementPointer()
	return FfiConverterBoolINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) C.int8_t {
		return C.uniffi_blazen_uniffi_fn_method_uniffiratedjsonldataset_is_empty(
			_pointer, _uniffiStatus)
	}))
}

// Number of rated examples in the dataset.
func (_self *UniffiRatedJsonlDataset) Len() uint64 {
	_pointer := _self.ffiObject.incrementPointer("*UniffiRatedJsonlDataset")
	defer _self.ffiObject.decrementPointer()
	return FfiConverterUint64INSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_method_uniffiratedjsonldataset_len(
			_pointer, _uniffiStatus)
	}))
}
func (object *UniffiRatedJsonlDataset) Destroy() {
	runtime.SetFinalizer(object, nil)
	object.ffiObject.destroy()
}

type FfiConverterUniffiRatedJsonlDataset struct{}

var FfiConverterUniffiRatedJsonlDatasetINSTANCE = FfiConverterUniffiRatedJsonlDataset{}

func (c FfiConverterUniffiRatedJsonlDataset) Lift(handle C.uint64_t) *UniffiRatedJsonlDataset {
	result := &UniffiRatedJsonlDataset{
		newFfiObject(
			handle,
			func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
				return C.uniffi_blazen_uniffi_fn_clone_uniffiratedjsonldataset(handle, status)
			},
			func(handle C.uint64_t, status *C.RustCallStatus) {
				C.uniffi_blazen_uniffi_fn_free_uniffiratedjsonldataset(handle, status)
			},
		),
	}
	runtime.SetFinalizer(result, (*UniffiRatedJsonlDataset).Destroy)
	return result
}

func (c FfiConverterUniffiRatedJsonlDataset) Read(reader io.Reader) *UniffiRatedJsonlDataset {
	return c.Lift(C.uint64_t(readUint64(reader)))
}

func (c FfiConverterUniffiRatedJsonlDataset) Lower(value *UniffiRatedJsonlDataset) C.uint64_t {
	// SAFETY (audited 2026-05-13): incrementPointer calls cloneFunction
	// which does Arc::clone on the Rust side, bumping the Rust refcount
	// independently of the Go-side callCounter. The defer below only
	// decrements the (redundant) Go counter; the returned handle survives
	// because the C caller owns its own Arc refcount via Arc::from_raw.
	handle := value.ffiObject.incrementPointer("*UniffiRatedJsonlDataset")
	defer value.ffiObject.decrementPointer()
	return handle
}

func (c FfiConverterUniffiRatedJsonlDataset) Write(writer io.Writer, value *UniffiRatedJsonlDataset) {
	writeUint64(writer, uint64(c.Lower(value)))
}

func LiftFromExternalUniffiRatedJsonlDataset(handle uint64) *UniffiRatedJsonlDataset {
	return FfiConverterUniffiRatedJsonlDatasetINSTANCE.Lift(C.uint64_t(handle))
}

func LowerToExternalUniffiRatedJsonlDataset(value *UniffiRatedJsonlDataset) uint64 {
	return uint64(FfiConverterUniffiRatedJsonlDatasetINSTANCE.Lower(value))
}

type FfiDestroyerUniffiRatedJsonlDataset struct{}

func (_ FfiDestroyerUniffiRatedJsonlDataset) Destroy(value *UniffiRatedJsonlDataset) {
	value.Destroy()
}

// A voice-conversion model.
//
// Construct via one of the per-backend factory functions (currently just
// [`new_rvc_model`], gated on `audio-vc-rvc`). Use the async
// [`convert_voice`](Self::convert_voice) method for one-shot rendering,
// [`list_target_voices`](Self::list_target_voices) /
// [`register_target_voice`](Self::register_target_voice) for voice
// management, or [`stream_convert_pcm_to_sink`] for chunk-level
// streaming.
type VcModelInterface interface {
	// Convert the source utterance at `input_audio_path` into the voice
	// of the registered target speaker `target_voice_id`.
	ConvertVoice(inputAudioPath string, targetVoiceId string) (VcResult, error)
	// Synchronous variant of [`convert_voice`](Self::convert_voice).
	ConvertVoiceBlocking(inputAudioPath string, targetVoiceId string) (VcResult, error)
	// List the target voices this backend can currently render.
	ListTargetVoices() ([]TargetVoice, error)
	// Synchronous variant of
	// [`list_target_voices`](Self::list_target_voices).
	ListTargetVoicesBlocking() ([]TargetVoice, error)
	// Register a new target voice from the reference utterance at
	// `reference_audio_path`.
	RegisterTargetVoice(voiceId string, referenceAudioPath string) error
	// Synchronous variant of
	// [`register_target_voice`](Self::register_target_voice).
	RegisterTargetVoiceBlocking(voiceId string, referenceAudioPath string) error
}

// A voice-conversion model.
//
// Construct via one of the per-backend factory functions (currently just
// [`new_rvc_model`], gated on `audio-vc-rvc`). Use the async
// [`convert_voice`](Self::convert_voice) method for one-shot rendering,
// [`list_target_voices`](Self::list_target_voices) /
// [`register_target_voice`](Self::register_target_voice) for voice
// management, or [`stream_convert_pcm_to_sink`] for chunk-level
// streaming.
type VcModel struct {
	ffiObject FfiObject
}

// Convert the source utterance at `input_audio_path` into the voice
// of the registered target speaker `target_voice_id`.
func (_self *VcModel) ConvertVoice(inputAudioPath string, targetVoiceId string) (VcResult, error) {
	_pointer := _self.ffiObject.incrementPointer("*VcModel")
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
		func(ffi RustBufferI) VcResult {
			return FfiConverterVcResultINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_vcmodel_convert_voice(
			_pointer, FfiConverterStringINSTANCE.Lower(inputAudioPath), FfiConverterStringINSTANCE.Lower(targetVoiceId)),
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

// Synchronous variant of [`convert_voice`](Self::convert_voice).
func (_self *VcModel) ConvertVoiceBlocking(inputAudioPath string, targetVoiceId string) (VcResult, error) {
	_pointer := _self.ffiObject.incrementPointer("*VcModel")
	defer _self.ffiObject.decrementPointer()
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) RustBufferI {
		return GoRustBuffer{
			inner: C.uniffi_blazen_uniffi_fn_method_vcmodel_convert_voice_blocking(
				_pointer, FfiConverterStringINSTANCE.Lower(inputAudioPath), FfiConverterStringINSTANCE.Lower(targetVoiceId), _uniffiStatus),
		}
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue VcResult
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterVcResultINSTANCE.Lift(_uniffiRV), nil
	}
}

// List the target voices this backend can currently render.
func (_self *VcModel) ListTargetVoices() ([]TargetVoice, error) {
	_pointer := _self.ffiObject.incrementPointer("*VcModel")
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
		func(ffi RustBufferI) []TargetVoice {
			return FfiConverterSequenceTargetVoiceINSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_method_vcmodel_list_target_voices(
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

// Synchronous variant of
// [`list_target_voices`](Self::list_target_voices).
func (_self *VcModel) ListTargetVoicesBlocking() ([]TargetVoice, error) {
	_pointer := _self.ffiObject.incrementPointer("*VcModel")
	defer _self.ffiObject.decrementPointer()
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) RustBufferI {
		return GoRustBuffer{
			inner: C.uniffi_blazen_uniffi_fn_method_vcmodel_list_target_voices_blocking(
				_pointer, _uniffiStatus),
		}
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue []TargetVoice
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterSequenceTargetVoiceINSTANCE.Lift(_uniffiRV), nil
	}
}

// Register a new target voice from the reference utterance at
// `reference_audio_path`.
func (_self *VcModel) RegisterTargetVoice(voiceId string, referenceAudioPath string) error {
	_pointer := _self.ffiObject.incrementPointer("*VcModel")
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
		C.uniffi_blazen_uniffi_fn_method_vcmodel_register_target_voice(
			_pointer, FfiConverterStringINSTANCE.Lower(voiceId), FfiConverterStringINSTANCE.Lower(referenceAudioPath)),
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

// Synchronous variant of
// [`register_target_voice`](Self::register_target_voice).
func (_self *VcModel) RegisterTargetVoiceBlocking(voiceId string, referenceAudioPath string) error {
	_pointer := _self.ffiObject.incrementPointer("*VcModel")
	defer _self.ffiObject.decrementPointer()
	_, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) bool {
		C.uniffi_blazen_uniffi_fn_method_vcmodel_register_target_voice_blocking(
			_pointer, FfiConverterStringINSTANCE.Lower(voiceId), FfiConverterStringINSTANCE.Lower(referenceAudioPath), _uniffiStatus)
		return false
	})
	return _uniffiErr.AsError()
}
func (object *VcModel) Destroy() {
	runtime.SetFinalizer(object, nil)
	object.ffiObject.destroy()
}

type FfiConverterVcModel struct{}

var FfiConverterVcModelINSTANCE = FfiConverterVcModel{}

func (c FfiConverterVcModel) Lift(handle C.uint64_t) *VcModel {
	result := &VcModel{
		newFfiObject(
			handle,
			func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
				return C.uniffi_blazen_uniffi_fn_clone_vcmodel(handle, status)
			},
			func(handle C.uint64_t, status *C.RustCallStatus) {
				C.uniffi_blazen_uniffi_fn_free_vcmodel(handle, status)
			},
		),
	}
	runtime.SetFinalizer(result, (*VcModel).Destroy)
	return result
}

func (c FfiConverterVcModel) Read(reader io.Reader) *VcModel {
	return c.Lift(C.uint64_t(readUint64(reader)))
}

func (c FfiConverterVcModel) Lower(value *VcModel) C.uint64_t {
	// SAFETY (audited 2026-05-13): incrementPointer calls cloneFunction
	// which does Arc::clone on the Rust side, bumping the Rust refcount
	// independently of the Go-side callCounter. The defer below only
	// decrements the (redundant) Go counter; the returned handle survives
	// because the C caller owns its own Arc refcount via Arc::from_raw.
	handle := value.ffiObject.incrementPointer("*VcModel")
	defer value.ffiObject.decrementPointer()
	return handle
}

func (c FfiConverterVcModel) Write(writer io.Writer, value *VcModel) {
	writeUint64(writer, uint64(c.Lower(value)))
}

func LiftFromExternalVcModel(handle uint64) *VcModel {
	return FfiConverterVcModelINSTANCE.Lift(C.uint64_t(handle))
}

func LowerToExternalVcModel(value *VcModel) uint64 {
	return uint64(FfiConverterVcModelINSTANCE.Lower(value))
}

type FfiDestroyerVcModel struct{}

func (_ FfiDestroyerVcModel) Destroy(value *VcModel) {
	value.Destroy()
}

// Sink for streaming voice-conversion output, implemented in foreign
// code.
//
// Symmetric to [`crate::compute_music::MusicStreamSink`] and
// [`crate::streaming::CompletionStreamSink`]: the streaming engine calls
// [`on_chunk`](Self::on_chunk) for each emitted chunk, then exactly one
// of [`on_done`](Self::on_done) or [`on_error`](Self::on_error).
// Implementations should treat the terminal callbacks as cleanup hooks
// (close channels, complete async iterators, signal flow completion).
type VcStreamSink interface {
	// Receive a single chunk from the streaming response.
	//
	// Returning an `Err` aborts the stream — the engine delivers the
	// error via [`on_error`](Self::on_error) and stops dispatching
	// further chunks.
	OnChunk(chunk VcChunk) error
	// Receive the terminal completion signal. Called exactly once at the
	// end of a successful stream.
	OnDone() error
	// Receive a fatal error from the stream. Called exactly once when
	// the stream fails midway (or fails to start at all).
	OnError(cause *BlazenError) error
}

// Sink for streaming voice-conversion output, implemented in foreign
// code.
//
// Symmetric to [`crate::compute_music::MusicStreamSink`] and
// [`crate::streaming::CompletionStreamSink`]: the streaming engine calls
// [`on_chunk`](Self::on_chunk) for each emitted chunk, then exactly one
// of [`on_done`](Self::on_done) or [`on_error`](Self::on_error).
// Implementations should treat the terminal callbacks as cleanup hooks
// (close channels, complete async iterators, signal flow completion).
type VcStreamSinkImpl struct {
	ffiObject FfiObject
}

// Receive a single chunk from the streaming response.
//
// Returning an `Err` aborts the stream — the engine delivers the
// error via [`on_error`](Self::on_error) and stops dispatching
// further chunks.
func (_self *VcStreamSinkImpl) OnChunk(chunk VcChunk) error {
	_pointer := _self.ffiObject.incrementPointer("VcStreamSink")
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
		C.uniffi_blazen_uniffi_fn_method_vcstreamsink_on_chunk(
			_pointer, FfiConverterVcChunkINSTANCE.Lower(chunk)),
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
// end of a successful stream.
func (_self *VcStreamSinkImpl) OnDone() error {
	_pointer := _self.ffiObject.incrementPointer("VcStreamSink")
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
		C.uniffi_blazen_uniffi_fn_method_vcstreamsink_on_done(
			_pointer),
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

// Receive a fatal error from the stream. Called exactly once when
// the stream fails midway (or fails to start at all).
func (_self *VcStreamSinkImpl) OnError(cause *BlazenError) error {
	_pointer := _self.ffiObject.incrementPointer("VcStreamSink")
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
		C.uniffi_blazen_uniffi_fn_method_vcstreamsink_on_error(
			_pointer, FfiConverterBlazenErrorINSTANCE.Lower(cause)),
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
func (object *VcStreamSinkImpl) Destroy() {
	runtime.SetFinalizer(object, nil)
	object.ffiObject.destroy()
}

type FfiConverterVcStreamSink struct {
	handleMap *concurrentHandleMap[VcStreamSink]
}

var FfiConverterVcStreamSinkINSTANCE = FfiConverterVcStreamSink{
	handleMap: newConcurrentHandleMap[VcStreamSink](),
}

func (c FfiConverterVcStreamSink) Lift(handle C.uint64_t) VcStreamSink {
	if uint64(handle)&1 == 0 {
		// Rust-generated handle (even), construct a new object wrapping the handle
		result := &VcStreamSinkImpl{
			newFfiObject(
				handle,
				func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
					return C.uniffi_blazen_uniffi_fn_clone_vcstreamsink(handle, status)
				},
				func(handle C.uint64_t, status *C.RustCallStatus) {
					C.uniffi_blazen_uniffi_fn_free_vcstreamsink(handle, status)
				},
			),
		}
		runtime.SetFinalizer(result, (*VcStreamSinkImpl).Destroy)
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

func (c FfiConverterVcStreamSink) Read(reader io.Reader) VcStreamSink {
	return c.Lift(C.uint64_t(readUint64(reader)))
}

func (c FfiConverterVcStreamSink) Lower(value VcStreamSink) C.uint64_t {
	// SAFETY (audited 2026-05-13): incrementPointer calls cloneFunction
	// which does Arc::clone on the Rust side, bumping the Rust refcount
	// independently of the Go-side callCounter. The defer below only
	// decrements the (redundant) Go counter; the returned handle survives
	// because the C caller owns its own Arc refcount via Arc::from_raw.
	if val, ok := value.(*VcStreamSinkImpl); ok {
		// Rust-backed object, clone the handle
		handle := val.ffiObject.incrementPointer("VcStreamSink")
		defer val.ffiObject.decrementPointer()
		return handle
	} else {
		// Go-backed object, insert into handle map
		return C.uint64_t(c.handleMap.insert(value))
	}
}

func (c FfiConverterVcStreamSink) Write(writer io.Writer, value VcStreamSink) {
	writeUint64(writer, uint64(c.Lower(value)))
}

func LiftFromExternalVcStreamSink(handle uint64) VcStreamSink {
	return FfiConverterVcStreamSinkINSTANCE.Lift(C.uint64_t(handle))
}

func LowerToExternalVcStreamSink(value VcStreamSink) uint64 {
	return uint64(FfiConverterVcStreamSinkINSTANCE.Lower(value))
}

type FfiDestroyerVcStreamSink struct{}

func (_ FfiDestroyerVcStreamSink) Destroy(value VcStreamSink) {
	if val, ok := value.(*VcStreamSinkImpl); ok {
		val.Destroy()
	}
}

//export blazen_uniffi_compute_vc_cgo_dispatchCallbackInterfaceVcStreamSinkMethod0
func blazen_uniffi_compute_vc_cgo_dispatchCallbackInterfaceVcStreamSinkMethod0(uniffiHandle C.uint64_t, chunk C.RustBuffer, uniffiFutureCallback C.UniffiForeignFutureCompleteVoid, uniffiCallbackData C.uint64_t, uniffiOutDroppedCallback *C.UniffiForeignFutureDroppedCallbackStruct) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterVcStreamSinkINSTANCE.handleMap.tryGet(handle)
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
				FfiConverterVcChunkINSTANCE.Lift(GoRustBuffer{
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

//export blazen_uniffi_compute_vc_cgo_dispatchCallbackInterfaceVcStreamSinkMethod1
func blazen_uniffi_compute_vc_cgo_dispatchCallbackInterfaceVcStreamSinkMethod1(uniffiHandle C.uint64_t, uniffiFutureCallback C.UniffiForeignFutureCompleteVoid, uniffiCallbackData C.uint64_t, uniffiOutDroppedCallback *C.UniffiForeignFutureDroppedCallbackStruct) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterVcStreamSinkINSTANCE.handleMap.tryGet(handle)
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
			uniffiObj.OnDone()

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

//export blazen_uniffi_compute_vc_cgo_dispatchCallbackInterfaceVcStreamSinkMethod2
func blazen_uniffi_compute_vc_cgo_dispatchCallbackInterfaceVcStreamSinkMethod2(uniffiHandle C.uint64_t, cause C.RustBuffer, uniffiFutureCallback C.UniffiForeignFutureCompleteVoid, uniffiCallbackData C.uint64_t, uniffiOutDroppedCallback *C.UniffiForeignFutureDroppedCallbackStruct) {
	handle := uint64(uniffiHandle)
	uniffiObj, ok := FfiConverterVcStreamSinkINSTANCE.handleMap.tryGet(handle)
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
					inner: cause,
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

var UniffiVTableCallbackInterfaceVcStreamSinkINSTANCE = C.UniffiVTableCallbackInterfaceVcStreamSink{
	uniffiFree:  (C.UniffiCallbackInterfaceFree)(C.blazen_uniffi_compute_vc_cgo_dispatchCallbackInterfaceVcStreamSinkFree),
	uniffiClone: (C.UniffiCallbackInterfaceClone)(C.blazen_uniffi_compute_vc_cgo_dispatchCallbackInterfaceVcStreamSinkClone),
	onChunk:     (C.UniffiCallbackInterfaceVcStreamSinkMethod0)(C.blazen_uniffi_compute_vc_cgo_dispatchCallbackInterfaceVcStreamSinkMethod0),
	onDone:      (C.UniffiCallbackInterfaceVcStreamSinkMethod1)(C.blazen_uniffi_compute_vc_cgo_dispatchCallbackInterfaceVcStreamSinkMethod1),
	onError:     (C.UniffiCallbackInterfaceVcStreamSinkMethod2)(C.blazen_uniffi_compute_vc_cgo_dispatchCallbackInterfaceVcStreamSinkMethod2),
}

//export blazen_uniffi_compute_vc_cgo_dispatchCallbackInterfaceVcStreamSinkFree
func blazen_uniffi_compute_vc_cgo_dispatchCallbackInterfaceVcStreamSinkFree(handle C.uint64_t) {
	FfiConverterVcStreamSinkINSTANCE.handleMap.remove(uint64(handle))
}

//export blazen_uniffi_compute_vc_cgo_dispatchCallbackInterfaceVcStreamSinkClone
func blazen_uniffi_compute_vc_cgo_dispatchCallbackInterfaceVcStreamSinkClone(handle C.uint64_t) C.uint64_t {
	val, ok := FfiConverterVcStreamSinkINSTANCE.handleMap.tryGet(uint64(handle))
	if !ok {
		panic(fmt.Errorf("no callback in handle map: %d", handle))
	}
	return C.uint64_t(FfiConverterVcStreamSinkINSTANCE.handleMap.insert(val))
}

func (c FfiConverterVcStreamSink) register() {
	C.uniffi_blazen_uniffi_fn_init_callback_vtable_vcstreamsink(&UniffiVTableCallbackInterfaceVcStreamSinkINSTANCE)
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
	// SAFETY (audited 2026-05-13): incrementPointer calls cloneFunction
	// which does Arc::clone on the Rust side, bumping the Rust refcount
	// independently of the Go-side callCounter. The defer below only
	// decrements the (redundant) Go counter; the returned handle survives
	// because the C caller owns its own Arc refcount via Arc::from_raw.
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
	// SAFETY (audited 2026-05-13): incrementPointer calls cloneFunction
	// which does Arc::clone on the Rust side, bumping the Rust refcount
	// independently of the Go-side callCounter. The defer below only
	// decrements the (redundant) Go counter; the returned handle survives
	// because the C caller owns its own Arc refcount via Arc::from_raw.
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

// Result returned by [`ForeignLocalModel::load_adapter`], mirroring
// [`blazen_llm::AdapterHandle`].
//
// `mount_strategy` is one of `"attached"`, `"rebuilt"`, `"merged"` — kept
// as a string discriminator so adding a new strategy to the upstream enum
// does not break the FFI contract.
type AdapterHandleRecord struct {
	AdapterId     string
	MemoryBytes   uint64
	MountStrategy string
}

func (r *AdapterHandleRecord) Destroy() {
	FfiDestroyerString{}.Destroy(r.AdapterId)
	FfiDestroyerUint64{}.Destroy(r.MemoryBytes)
	FfiDestroyerString{}.Destroy(r.MountStrategy)
}

type FfiConverterAdapterHandleRecord struct{}

var FfiConverterAdapterHandleRecordINSTANCE = FfiConverterAdapterHandleRecord{}

func (c FfiConverterAdapterHandleRecord) Lift(rb RustBufferI) AdapterHandleRecord {
	return LiftFromRustBuffer[AdapterHandleRecord](c, rb)
}

func (c FfiConverterAdapterHandleRecord) Read(reader io.Reader) AdapterHandleRecord {
	return AdapterHandleRecord{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterUint64INSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
	}
}

func (c FfiConverterAdapterHandleRecord) Lower(value AdapterHandleRecord) C.RustBuffer {
	return LowerIntoRustBuffer[AdapterHandleRecord](c, value)
}

func (c FfiConverterAdapterHandleRecord) LowerExternal(value AdapterHandleRecord) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[AdapterHandleRecord](c, value))
}

func (c FfiConverterAdapterHandleRecord) Write(writer io.Writer, value AdapterHandleRecord) {
	FfiConverterStringINSTANCE.Write(writer, value.AdapterId)
	FfiConverterUint64INSTANCE.Write(writer, value.MemoryBytes)
	FfiConverterStringINSTANCE.Write(writer, value.MountStrategy)
}

type FfiDestroyerAdapterHandleRecord struct{}

func (_ FfiDestroyerAdapterHandleRecord) Destroy(value AdapterHandleRecord) {
	value.Destroy()
}

// Adapter mount options handed to [`ForeignLocalModel::load_adapter`].
//
// Mirrors [`blazen_llm::AdapterOptions`] but lives as a UniFFI Record so
// foreign code can construct it natively.
type AdapterOptionsRecord struct {
	AdapterId string
	Scale     float32
}

func (r *AdapterOptionsRecord) Destroy() {
	FfiDestroyerString{}.Destroy(r.AdapterId)
	FfiDestroyerFloat32{}.Destroy(r.Scale)
}

type FfiConverterAdapterOptionsRecord struct{}

var FfiConverterAdapterOptionsRecordINSTANCE = FfiConverterAdapterOptionsRecord{}

func (c FfiConverterAdapterOptionsRecord) Lift(rb RustBufferI) AdapterOptionsRecord {
	return LiftFromRustBuffer[AdapterOptionsRecord](c, rb)
}

func (c FfiConverterAdapterOptionsRecord) Read(reader io.Reader) AdapterOptionsRecord {
	return AdapterOptionsRecord{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterFloat32INSTANCE.Read(reader),
	}
}

func (c FfiConverterAdapterOptionsRecord) Lower(value AdapterOptionsRecord) C.RustBuffer {
	return LowerIntoRustBuffer[AdapterOptionsRecord](c, value)
}

func (c FfiConverterAdapterOptionsRecord) LowerExternal(value AdapterOptionsRecord) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[AdapterOptionsRecord](c, value))
}

func (c FfiConverterAdapterOptionsRecord) Write(writer io.Writer, value AdapterOptionsRecord) {
	FfiConverterStringINSTANCE.Write(writer, value.AdapterId)
	FfiConverterFloat32INSTANCE.Write(writer, value.Scale)
}

type FfiDestroyerAdapterOptionsRecord struct{}

func (_ FfiDestroyerAdapterOptionsRecord) Destroy(value AdapterOptionsRecord) {
	value.Destroy()
}

// Snapshot of a single mounted adapter — wire form of
// [`blazen_llm::AdapterStatus`].
type AdapterStatusRecord struct {
	AdapterId   string
	Scale       float32
	SourceDir   string
	MemoryBytes uint64
}

func (r *AdapterStatusRecord) Destroy() {
	FfiDestroyerString{}.Destroy(r.AdapterId)
	FfiDestroyerFloat32{}.Destroy(r.Scale)
	FfiDestroyerString{}.Destroy(r.SourceDir)
	FfiDestroyerUint64{}.Destroy(r.MemoryBytes)
}

type FfiConverterAdapterStatusRecord struct{}

var FfiConverterAdapterStatusRecordINSTANCE = FfiConverterAdapterStatusRecord{}

func (c FfiConverterAdapterStatusRecord) Lift(rb RustBufferI) AdapterStatusRecord {
	return LiftFromRustBuffer[AdapterStatusRecord](c, rb)
}

func (c FfiConverterAdapterStatusRecord) Read(reader io.Reader) AdapterStatusRecord {
	return AdapterStatusRecord{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterFloat32INSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterUint64INSTANCE.Read(reader),
	}
}

func (c FfiConverterAdapterStatusRecord) Lower(value AdapterStatusRecord) C.RustBuffer {
	return LowerIntoRustBuffer[AdapterStatusRecord](c, value)
}

func (c FfiConverterAdapterStatusRecord) LowerExternal(value AdapterStatusRecord) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[AdapterStatusRecord](c, value))
}

func (c FfiConverterAdapterStatusRecord) Write(writer io.Writer, value AdapterStatusRecord) {
	FfiConverterStringINSTANCE.Write(writer, value.AdapterId)
	FfiConverterFloat32INSTANCE.Write(writer, value.Scale)
	FfiConverterStringINSTANCE.Write(writer, value.SourceDir)
	FfiConverterUint64INSTANCE.Write(writer, value.MemoryBytes)
}

type FfiDestroyerAdapterStatusRecord struct{}

func (_ FfiDestroyerAdapterStatusRecord) Destroy(value AdapterStatusRecord) {
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

// Bundle of admission-policy fields for a worker.
//
// `max_in_flight` is meaningful when `mode == Fixed`, `total_mb` when
// `mode == VramBudget`; both fields are ignored when `mode == Reactive`.
// Either may be omitted to fall back to upstream defaults
// (`Fixed { max_in_flight: 1 }`, `VramBudget { max_vram_mb: 0 }`).
type ControlPlaneAdmission struct {
	Mode        ControlPlaneAdmissionMode
	MaxInFlight *uint32
	TotalMb     *uint32
}

func (r *ControlPlaneAdmission) Destroy() {
	FfiDestroyerControlPlaneAdmissionMode{}.Destroy(r.Mode)
	FfiDestroyerOptionalUint32{}.Destroy(r.MaxInFlight)
	FfiDestroyerOptionalUint32{}.Destroy(r.TotalMb)
}

type FfiConverterControlPlaneAdmission struct{}

var FfiConverterControlPlaneAdmissionINSTANCE = FfiConverterControlPlaneAdmission{}

func (c FfiConverterControlPlaneAdmission) Lift(rb RustBufferI) ControlPlaneAdmission {
	return LiftFromRustBuffer[ControlPlaneAdmission](c, rb)
}

func (c FfiConverterControlPlaneAdmission) Read(reader io.Reader) ControlPlaneAdmission {
	return ControlPlaneAdmission{
		FfiConverterControlPlaneAdmissionModeINSTANCE.Read(reader),
		FfiConverterOptionalUint32INSTANCE.Read(reader),
		FfiConverterOptionalUint32INSTANCE.Read(reader),
	}
}

func (c FfiConverterControlPlaneAdmission) Lower(value ControlPlaneAdmission) C.RustBuffer {
	return LowerIntoRustBuffer[ControlPlaneAdmission](c, value)
}

func (c FfiConverterControlPlaneAdmission) LowerExternal(value ControlPlaneAdmission) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[ControlPlaneAdmission](c, value))
}

func (c FfiConverterControlPlaneAdmission) Write(writer io.Writer, value ControlPlaneAdmission) {
	FfiConverterControlPlaneAdmissionModeINSTANCE.Write(writer, value.Mode)
	FfiConverterOptionalUint32INSTANCE.Write(writer, value.MaxInFlight)
	FfiConverterOptionalUint32INSTANCE.Write(writer, value.TotalMb)
}

type FfiDestroyerControlPlaneAdmission struct{}

func (_ FfiDestroyerControlPlaneAdmission) Destroy(value ControlPlaneAdmission) {
	value.Destroy()
}

// Foreign-facing run event.
//
// `data_json` is the upstream `data: serde_json::Value` serialized to a
// JSON string for transport across the UniFFI boundary.
type ControlPlaneRunEvent struct {
	RunId       string
	EventType   string
	DataJson    string
	TimestampMs uint64
}

func (r *ControlPlaneRunEvent) Destroy() {
	FfiDestroyerString{}.Destroy(r.RunId)
	FfiDestroyerString{}.Destroy(r.EventType)
	FfiDestroyerString{}.Destroy(r.DataJson)
	FfiDestroyerUint64{}.Destroy(r.TimestampMs)
}

type FfiConverterControlPlaneRunEvent struct{}

var FfiConverterControlPlaneRunEventINSTANCE = FfiConverterControlPlaneRunEvent{}

func (c FfiConverterControlPlaneRunEvent) Lift(rb RustBufferI) ControlPlaneRunEvent {
	return LiftFromRustBuffer[ControlPlaneRunEvent](c, rb)
}

func (c FfiConverterControlPlaneRunEvent) Read(reader io.Reader) ControlPlaneRunEvent {
	return ControlPlaneRunEvent{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterUint64INSTANCE.Read(reader),
	}
}

func (c FfiConverterControlPlaneRunEvent) Lower(value ControlPlaneRunEvent) C.RustBuffer {
	return LowerIntoRustBuffer[ControlPlaneRunEvent](c, value)
}

func (c FfiConverterControlPlaneRunEvent) LowerExternal(value ControlPlaneRunEvent) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[ControlPlaneRunEvent](c, value))
}

func (c FfiConverterControlPlaneRunEvent) Write(writer io.Writer, value ControlPlaneRunEvent) {
	FfiConverterStringINSTANCE.Write(writer, value.RunId)
	FfiConverterStringINSTANCE.Write(writer, value.EventType)
	FfiConverterStringINSTANCE.Write(writer, value.DataJson)
	FfiConverterUint64INSTANCE.Write(writer, value.TimestampMs)
}

type FfiDestroyerControlPlaneRunEvent struct{}

func (_ FfiDestroyerControlPlaneRunEvent) Destroy(value ControlPlaneRunEvent) {
	value.Destroy()
}

// Foreign-facing snapshot of a workflow run.
//
// `run_id` is the canonical UUID string (`"550e8400-e29b-41d4-a716-446655440000"`);
// `output_json` and `error` are flattened from the upstream snapshot's
// `output: Option<serde_json::Value>` / `error: Option<String>` fields.
type ControlPlaneRunStateSnapshot struct {
	RunId         string
	Status        ControlPlaneRunStatus
	StartedAtMs   uint64
	CompletedAtMs *uint64
	AssignedTo    *string
	LastEventAtMs *uint64
	OutputJson    *string
	Error         *string
}

func (r *ControlPlaneRunStateSnapshot) Destroy() {
	FfiDestroyerString{}.Destroy(r.RunId)
	FfiDestroyerControlPlaneRunStatus{}.Destroy(r.Status)
	FfiDestroyerUint64{}.Destroy(r.StartedAtMs)
	FfiDestroyerOptionalUint64{}.Destroy(r.CompletedAtMs)
	FfiDestroyerOptionalString{}.Destroy(r.AssignedTo)
	FfiDestroyerOptionalUint64{}.Destroy(r.LastEventAtMs)
	FfiDestroyerOptionalString{}.Destroy(r.OutputJson)
	FfiDestroyerOptionalString{}.Destroy(r.Error)
}

type FfiConverterControlPlaneRunStateSnapshot struct{}

var FfiConverterControlPlaneRunStateSnapshotINSTANCE = FfiConverterControlPlaneRunStateSnapshot{}

func (c FfiConverterControlPlaneRunStateSnapshot) Lift(rb RustBufferI) ControlPlaneRunStateSnapshot {
	return LiftFromRustBuffer[ControlPlaneRunStateSnapshot](c, rb)
}

func (c FfiConverterControlPlaneRunStateSnapshot) Read(reader io.Reader) ControlPlaneRunStateSnapshot {
	return ControlPlaneRunStateSnapshot{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterControlPlaneRunStatusINSTANCE.Read(reader),
		FfiConverterUint64INSTANCE.Read(reader),
		FfiConverterOptionalUint64INSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterOptionalUint64INSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
	}
}

func (c FfiConverterControlPlaneRunStateSnapshot) Lower(value ControlPlaneRunStateSnapshot) C.RustBuffer {
	return LowerIntoRustBuffer[ControlPlaneRunStateSnapshot](c, value)
}

func (c FfiConverterControlPlaneRunStateSnapshot) LowerExternal(value ControlPlaneRunStateSnapshot) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[ControlPlaneRunStateSnapshot](c, value))
}

func (c FfiConverterControlPlaneRunStateSnapshot) Write(writer io.Writer, value ControlPlaneRunStateSnapshot) {
	FfiConverterStringINSTANCE.Write(writer, value.RunId)
	FfiConverterControlPlaneRunStatusINSTANCE.Write(writer, value.Status)
	FfiConverterUint64INSTANCE.Write(writer, value.StartedAtMs)
	FfiConverterOptionalUint64INSTANCE.Write(writer, value.CompletedAtMs)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.AssignedTo)
	FfiConverterOptionalUint64INSTANCE.Write(writer, value.LastEventAtMs)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.OutputJson)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.Error)
}

type FfiDestroyerControlPlaneRunStateSnapshot struct{}

func (_ FfiDestroyerControlPlaneRunStateSnapshot) Destroy(value ControlPlaneRunStateSnapshot) {
	value.Destroy()
}

// Foreign-facing workflow submission request.
//
// Mirrors [`CoreSubmitWorkflowRequest`] except `input_json` carries the
// initial input as a JSON-encoded string and `resource_hint` is omitted
// (the UniFFI surface today targets Fixed/Reactive admission only;
// VramBudget callers should target the native crate directly).
type ControlPlaneSubmitRequest struct {
	WorkflowName    string
	InputJson       string
	WorkflowVersion *uint32
	RequiredTags    []string
	IdempotencyKey  *string
	DeadlineMs      *uint64
	WaitForWorker   bool
}

func (r *ControlPlaneSubmitRequest) Destroy() {
	FfiDestroyerString{}.Destroy(r.WorkflowName)
	FfiDestroyerString{}.Destroy(r.InputJson)
	FfiDestroyerOptionalUint32{}.Destroy(r.WorkflowVersion)
	FfiDestroyerSequenceString{}.Destroy(r.RequiredTags)
	FfiDestroyerOptionalString{}.Destroy(r.IdempotencyKey)
	FfiDestroyerOptionalUint64{}.Destroy(r.DeadlineMs)
	FfiDestroyerBool{}.Destroy(r.WaitForWorker)
}

type FfiConverterControlPlaneSubmitRequest struct{}

var FfiConverterControlPlaneSubmitRequestINSTANCE = FfiConverterControlPlaneSubmitRequest{}

func (c FfiConverterControlPlaneSubmitRequest) Lift(rb RustBufferI) ControlPlaneSubmitRequest {
	return LiftFromRustBuffer[ControlPlaneSubmitRequest](c, rb)
}

func (c FfiConverterControlPlaneSubmitRequest) Read(reader io.Reader) ControlPlaneSubmitRequest {
	return ControlPlaneSubmitRequest{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterOptionalUint32INSTANCE.Read(reader),
		FfiConverterSequenceStringINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterOptionalUint64INSTANCE.Read(reader),
		FfiConverterBoolINSTANCE.Read(reader),
	}
}

func (c FfiConverterControlPlaneSubmitRequest) Lower(value ControlPlaneSubmitRequest) C.RustBuffer {
	return LowerIntoRustBuffer[ControlPlaneSubmitRequest](c, value)
}

func (c FfiConverterControlPlaneSubmitRequest) LowerExternal(value ControlPlaneSubmitRequest) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[ControlPlaneSubmitRequest](c, value))
}

func (c FfiConverterControlPlaneSubmitRequest) Write(writer io.Writer, value ControlPlaneSubmitRequest) {
	FfiConverterStringINSTANCE.Write(writer, value.WorkflowName)
	FfiConverterStringINSTANCE.Write(writer, value.InputJson)
	FfiConverterOptionalUint32INSTANCE.Write(writer, value.WorkflowVersion)
	FfiConverterSequenceStringINSTANCE.Write(writer, value.RequiredTags)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.IdempotencyKey)
	FfiConverterOptionalUint64INSTANCE.Write(writer, value.DeadlineMs)
	FfiConverterBoolINSTANCE.Write(writer, value.WaitForWorker)
}

type FfiDestroyerControlPlaneSubmitRequest struct{}

func (_ FfiDestroyerControlPlaneSubmitRequest) Destroy(value ControlPlaneSubmitRequest) {
	value.Destroy()
}

// Typed capability a worker advertises to the control plane.
//
// `kind` follows the convention `"workflow:<name>"` /
// `"step:<name>"` / `"provider:<id>"` / `"tag:<key>=<value>"`.
// `version` lets the control plane gate routing on schema changes.
type ControlPlaneWorkerCapability struct {
	Kind    string
	Version uint32
}

func (r *ControlPlaneWorkerCapability) Destroy() {
	FfiDestroyerString{}.Destroy(r.Kind)
	FfiDestroyerUint32{}.Destroy(r.Version)
}

type FfiConverterControlPlaneWorkerCapability struct{}

var FfiConverterControlPlaneWorkerCapabilityINSTANCE = FfiConverterControlPlaneWorkerCapability{}

func (c FfiConverterControlPlaneWorkerCapability) Lift(rb RustBufferI) ControlPlaneWorkerCapability {
	return LiftFromRustBuffer[ControlPlaneWorkerCapability](c, rb)
}

func (c FfiConverterControlPlaneWorkerCapability) Read(reader io.Reader) ControlPlaneWorkerCapability {
	return ControlPlaneWorkerCapability{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterUint32INSTANCE.Read(reader),
	}
}

func (c FfiConverterControlPlaneWorkerCapability) Lower(value ControlPlaneWorkerCapability) C.RustBuffer {
	return LowerIntoRustBuffer[ControlPlaneWorkerCapability](c, value)
}

func (c FfiConverterControlPlaneWorkerCapability) LowerExternal(value ControlPlaneWorkerCapability) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[ControlPlaneWorkerCapability](c, value))
}

func (c FfiConverterControlPlaneWorkerCapability) Write(writer io.Writer, value ControlPlaneWorkerCapability) {
	FfiConverterStringINSTANCE.Write(writer, value.Kind)
	FfiConverterUint32INSTANCE.Write(writer, value.Version)
}

type FfiDestroyerControlPlaneWorkerCapability struct{}

func (_ FfiDestroyerControlPlaneWorkerCapability) Destroy(value ControlPlaneWorkerCapability) {
	value.Destroy()
}

// Foreign-facing summary of a connected worker.
//
// Upstream [`CoreWorkerInfo`] carries an `admission_snapshot` and an
// `admission` field; this surface omits the snapshot (foreign callers
// who need it should query the control plane directly) and flattens
// `tags` from a `BTreeMap` to a [`HashMap`] for UniFFI compatibility.
type ControlPlaneWorkerInfo struct {
	NodeId        string
	Capabilities  []ControlPlaneWorkerCapability
	Tags          map[string]string
	InFlight      uint32
	ConnectedAtMs uint64
}

func (r *ControlPlaneWorkerInfo) Destroy() {
	FfiDestroyerString{}.Destroy(r.NodeId)
	FfiDestroyerSequenceControlPlaneWorkerCapability{}.Destroy(r.Capabilities)
	FfiDestroyerMapStringString{}.Destroy(r.Tags)
	FfiDestroyerUint32{}.Destroy(r.InFlight)
	FfiDestroyerUint64{}.Destroy(r.ConnectedAtMs)
}

type FfiConverterControlPlaneWorkerInfo struct{}

var FfiConverterControlPlaneWorkerInfoINSTANCE = FfiConverterControlPlaneWorkerInfo{}

func (c FfiConverterControlPlaneWorkerInfo) Lift(rb RustBufferI) ControlPlaneWorkerInfo {
	return LiftFromRustBuffer[ControlPlaneWorkerInfo](c, rb)
}

func (c FfiConverterControlPlaneWorkerInfo) Read(reader io.Reader) ControlPlaneWorkerInfo {
	return ControlPlaneWorkerInfo{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterSequenceControlPlaneWorkerCapabilityINSTANCE.Read(reader),
		FfiConverterMapStringStringINSTANCE.Read(reader),
		FfiConverterUint32INSTANCE.Read(reader),
		FfiConverterUint64INSTANCE.Read(reader),
	}
}

func (c FfiConverterControlPlaneWorkerInfo) Lower(value ControlPlaneWorkerInfo) C.RustBuffer {
	return LowerIntoRustBuffer[ControlPlaneWorkerInfo](c, value)
}

func (c FfiConverterControlPlaneWorkerInfo) LowerExternal(value ControlPlaneWorkerInfo) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[ControlPlaneWorkerInfo](c, value))
}

func (c FfiConverterControlPlaneWorkerInfo) Write(writer io.Writer, value ControlPlaneWorkerInfo) {
	FfiConverterStringINSTANCE.Write(writer, value.NodeId)
	FfiConverterSequenceControlPlaneWorkerCapabilityINSTANCE.Write(writer, value.Capabilities)
	FfiConverterMapStringStringINSTANCE.Write(writer, value.Tags)
	FfiConverterUint32INSTANCE.Write(writer, value.InFlight)
	FfiConverterUint64INSTANCE.Write(writer, value.ConnectedAtMs)
}

type FfiDestroyerControlPlaneWorkerInfo struct{}

func (_ FfiDestroyerControlPlaneWorkerInfo) Destroy(value ControlPlaneWorkerInfo) {
	value.Destroy()
}

// Configuration for distributed (ring-AllReduce) training.
//
// `rank` is the 0-indexed rank of this worker; `world_size` is the
// total number of workers. `peers` is the ordered list of
// `"host:port"` gRPC endpoints — one entry per rank. `master_addr`
// + `master_port` identify the bootstrap node (typically the host
// part of `peers[0]`).
type DistributedConfigRecord struct {
	Rank       uint32
	WorldSize  uint32
	Peers      []string
	MasterAddr string
	MasterPort uint16
}

func (r *DistributedConfigRecord) Destroy() {
	FfiDestroyerUint32{}.Destroy(r.Rank)
	FfiDestroyerUint32{}.Destroy(r.WorldSize)
	FfiDestroyerSequenceString{}.Destroy(r.Peers)
	FfiDestroyerString{}.Destroy(r.MasterAddr)
	FfiDestroyerUint16{}.Destroy(r.MasterPort)
}

type FfiConverterDistributedConfigRecord struct{}

var FfiConverterDistributedConfigRecordINSTANCE = FfiConverterDistributedConfigRecord{}

func (c FfiConverterDistributedConfigRecord) Lift(rb RustBufferI) DistributedConfigRecord {
	return LiftFromRustBuffer[DistributedConfigRecord](c, rb)
}

func (c FfiConverterDistributedConfigRecord) Read(reader io.Reader) DistributedConfigRecord {
	return DistributedConfigRecord{
		FfiConverterUint32INSTANCE.Read(reader),
		FfiConverterUint32INSTANCE.Read(reader),
		FfiConverterSequenceStringINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterUint16INSTANCE.Read(reader),
	}
}

func (c FfiConverterDistributedConfigRecord) Lower(value DistributedConfigRecord) C.RustBuffer {
	return LowerIntoRustBuffer[DistributedConfigRecord](c, value)
}

func (c FfiConverterDistributedConfigRecord) LowerExternal(value DistributedConfigRecord) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[DistributedConfigRecord](c, value))
}

func (c FfiConverterDistributedConfigRecord) Write(writer io.Writer, value DistributedConfigRecord) {
	FfiConverterUint32INSTANCE.Write(writer, value.Rank)
	FfiConverterUint32INSTANCE.Write(writer, value.WorldSize)
	FfiConverterSequenceStringINSTANCE.Write(writer, value.Peers)
	FfiConverterStringINSTANCE.Write(writer, value.MasterAddr)
	FfiConverterUint16INSTANCE.Write(writer, value.MasterPort)
}

type FfiDestroyerDistributedConfigRecord struct{}

func (_ FfiDestroyerDistributedConfigRecord) Destroy(value DistributedConfigRecord) {
	value.Destroy()
}

// Direct Preference Optimization (DPO) configuration.
//
// Requires a frozen reference model. If `reference_model_repo` is
// `None`, the trainer reuses `core.base_model_repo`.
type DpoConfigRecord struct {
	Core                   TrainCoreConfigRecord
	Lora                   LoraConfigRecord
	Beta                   float32
	LabelSmoothing         float32
	ReferenceModelRepo     *string
	ReferenceModelRevision *string
}

func (r *DpoConfigRecord) Destroy() {
	FfiDestroyerTrainCoreConfigRecord{}.Destroy(r.Core)
	FfiDestroyerLoraConfigRecord{}.Destroy(r.Lora)
	FfiDestroyerFloat32{}.Destroy(r.Beta)
	FfiDestroyerFloat32{}.Destroy(r.LabelSmoothing)
	FfiDestroyerOptionalString{}.Destroy(r.ReferenceModelRepo)
	FfiDestroyerOptionalString{}.Destroy(r.ReferenceModelRevision)
}

type FfiConverterDpoConfigRecord struct{}

var FfiConverterDpoConfigRecordINSTANCE = FfiConverterDpoConfigRecord{}

func (c FfiConverterDpoConfigRecord) Lift(rb RustBufferI) DpoConfigRecord {
	return LiftFromRustBuffer[DpoConfigRecord](c, rb)
}

func (c FfiConverterDpoConfigRecord) Read(reader io.Reader) DpoConfigRecord {
	return DpoConfigRecord{
		FfiConverterTrainCoreConfigRecordINSTANCE.Read(reader),
		FfiConverterLoraConfigRecordINSTANCE.Read(reader),
		FfiConverterFloat32INSTANCE.Read(reader),
		FfiConverterFloat32INSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
	}
}

func (c FfiConverterDpoConfigRecord) Lower(value DpoConfigRecord) C.RustBuffer {
	return LowerIntoRustBuffer[DpoConfigRecord](c, value)
}

func (c FfiConverterDpoConfigRecord) LowerExternal(value DpoConfigRecord) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[DpoConfigRecord](c, value))
}

func (c FfiConverterDpoConfigRecord) Write(writer io.Writer, value DpoConfigRecord) {
	FfiConverterTrainCoreConfigRecordINSTANCE.Write(writer, value.Core)
	FfiConverterLoraConfigRecordINSTANCE.Write(writer, value.Lora)
	FfiConverterFloat32INSTANCE.Write(writer, value.Beta)
	FfiConverterFloat32INSTANCE.Write(writer, value.LabelSmoothing)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.ReferenceModelRepo)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.ReferenceModelRevision)
}

type FfiDestroyerDpoConfigRecord struct{}

func (_ FfiDestroyerDpoConfigRecord) Destroy(value DpoConfigRecord) {
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

// Full fine-tune configuration (no LoRA — every parameter trains).
//
// `gradient_checkpointing = true` is accepted for forward compatibility
// but the trainer rejects it at init time with
// [`BlazenError::Validation`] — candle 0.10.2 has no activation-
// checkpointing primitive.
type FullFineTuneConfigRecord struct {
	Core                  TrainCoreConfigRecord
	GradientCheckpointing bool
}

func (r *FullFineTuneConfigRecord) Destroy() {
	FfiDestroyerTrainCoreConfigRecord{}.Destroy(r.Core)
	FfiDestroyerBool{}.Destroy(r.GradientCheckpointing)
}

type FfiConverterFullFineTuneConfigRecord struct{}

var FfiConverterFullFineTuneConfigRecordINSTANCE = FfiConverterFullFineTuneConfigRecord{}

func (c FfiConverterFullFineTuneConfigRecord) Lift(rb RustBufferI) FullFineTuneConfigRecord {
	return LiftFromRustBuffer[FullFineTuneConfigRecord](c, rb)
}

func (c FfiConverterFullFineTuneConfigRecord) Read(reader io.Reader) FullFineTuneConfigRecord {
	return FullFineTuneConfigRecord{
		FfiConverterTrainCoreConfigRecordINSTANCE.Read(reader),
		FfiConverterBoolINSTANCE.Read(reader),
	}
}

func (c FfiConverterFullFineTuneConfigRecord) Lower(value FullFineTuneConfigRecord) C.RustBuffer {
	return LowerIntoRustBuffer[FullFineTuneConfigRecord](c, value)
}

func (c FfiConverterFullFineTuneConfigRecord) LowerExternal(value FullFineTuneConfigRecord) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[FullFineTuneConfigRecord](c, value))
}

func (c FfiConverterFullFineTuneConfigRecord) Write(writer io.Writer, value FullFineTuneConfigRecord) {
	FfiConverterTrainCoreConfigRecordINSTANCE.Write(writer, value.Core)
	FfiConverterBoolINSTANCE.Write(writer, value.GradientCheckpointing)
}

type FfiDestroyerFullFineTuneConfigRecord struct{}

func (_ FfiDestroyerFullFineTuneConfigRecord) Destroy(value FullFineTuneConfigRecord) {
	value.Destroy()
}

// On-disk descriptor returned by [`UniffiModelManager::fine_tune`].
//
// Unlike [`TrainedAdapterRecord`], no PEFT adapter is written — the
// entire model's weights are saved to `output_dir` directly.
type FullFineTuneResultRecord struct {
	OutputDir      string
	FinalLoss      float32
	StepsCompleted uint64
}

func (r *FullFineTuneResultRecord) Destroy() {
	FfiDestroyerString{}.Destroy(r.OutputDir)
	FfiDestroyerFloat32{}.Destroy(r.FinalLoss)
	FfiDestroyerUint64{}.Destroy(r.StepsCompleted)
}

type FfiConverterFullFineTuneResultRecord struct{}

var FfiConverterFullFineTuneResultRecordINSTANCE = FfiConverterFullFineTuneResultRecord{}

func (c FfiConverterFullFineTuneResultRecord) Lift(rb RustBufferI) FullFineTuneResultRecord {
	return LiftFromRustBuffer[FullFineTuneResultRecord](c, rb)
}

func (c FfiConverterFullFineTuneResultRecord) Read(reader io.Reader) FullFineTuneResultRecord {
	return FullFineTuneResultRecord{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterFloat32INSTANCE.Read(reader),
		FfiConverterUint64INSTANCE.Read(reader),
	}
}

func (c FfiConverterFullFineTuneResultRecord) Lower(value FullFineTuneResultRecord) C.RustBuffer {
	return LowerIntoRustBuffer[FullFineTuneResultRecord](c, value)
}

func (c FfiConverterFullFineTuneResultRecord) LowerExternal(value FullFineTuneResultRecord) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[FullFineTuneResultRecord](c, value))
}

func (c FfiConverterFullFineTuneResultRecord) Write(writer io.Writer, value FullFineTuneResultRecord) {
	FfiConverterStringINSTANCE.Write(writer, value.OutputDir)
	FfiConverterFloat32INSTANCE.Write(writer, value.FinalLoss)
	FfiConverterUint64INSTANCE.Write(writer, value.StepsCompleted)
}

type FfiDestroyerFullFineTuneResultRecord struct{}

func (_ FfiDestroyerFullFineTuneResultRecord) Destroy(value FullFineTuneResultRecord) {
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

// Caller-supplied options for [`UniffiModelManager::load_from_hf`].
//
// Mirrors [`blazen_manager::hf_loader::HfLoadOptions`]; every field is
// optional. `pool` is a label (`"cpu"`, `"gpu"`, `"gpu:N"`) and is parsed
// by `parse_pool_label`.
type HfLoadOptionsRecord struct {
	// Force a specific backend; skips engine inference but still probes
	// the repo for memory sizing.
	BackendHint *BackendHintEnum
	// Git revision (branch, tag, or commit sha). Defaults to the repo's
	// default branch.
	Revision *string
	// Hugging Face access token. When omitted, falls back to the
	// `HF_TOKEN` environment variable, then to anonymous access.
	HfToken *string
	// Override the on-disk cache directory used by `hf-hub`.
	CacheDir *string
	// Device specifier forwarded to the chosen provider (`"cpu"`,
	// `"cuda:0"`, `"metal"`, …).
	Device *string
	// Explicit GGUF filename for repos that ship multiple quantizations.
	GgufFile *string
	// Override the auto-derived memory estimate, in bytes.
	MemoryEstimateBytes *uint64
	// Pool label (`"cpu"`, `"gpu"`, `"gpu:N"`). Defaults to `"cpu"`.
	Pool *string
}

func (r *HfLoadOptionsRecord) Destroy() {
	FfiDestroyerOptionalBackendHintEnum{}.Destroy(r.BackendHint)
	FfiDestroyerOptionalString{}.Destroy(r.Revision)
	FfiDestroyerOptionalString{}.Destroy(r.HfToken)
	FfiDestroyerOptionalString{}.Destroy(r.CacheDir)
	FfiDestroyerOptionalString{}.Destroy(r.Device)
	FfiDestroyerOptionalString{}.Destroy(r.GgufFile)
	FfiDestroyerOptionalUint64{}.Destroy(r.MemoryEstimateBytes)
	FfiDestroyerOptionalString{}.Destroy(r.Pool)
}

type FfiConverterHfLoadOptionsRecord struct{}

var FfiConverterHfLoadOptionsRecordINSTANCE = FfiConverterHfLoadOptionsRecord{}

func (c FfiConverterHfLoadOptionsRecord) Lift(rb RustBufferI) HfLoadOptionsRecord {
	return LiftFromRustBuffer[HfLoadOptionsRecord](c, rb)
}

func (c FfiConverterHfLoadOptionsRecord) Read(reader io.Reader) HfLoadOptionsRecord {
	return HfLoadOptionsRecord{
		FfiConverterOptionalBackendHintEnumINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterOptionalUint64INSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
	}
}

func (c FfiConverterHfLoadOptionsRecord) Lower(value HfLoadOptionsRecord) C.RustBuffer {
	return LowerIntoRustBuffer[HfLoadOptionsRecord](c, value)
}

func (c FfiConverterHfLoadOptionsRecord) LowerExternal(value HfLoadOptionsRecord) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[HfLoadOptionsRecord](c, value))
}

func (c FfiConverterHfLoadOptionsRecord) Write(writer io.Writer, value HfLoadOptionsRecord) {
	FfiConverterOptionalBackendHintEnumINSTANCE.Write(writer, value.BackendHint)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.Revision)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.HfToken)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.CacheDir)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.Device)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.GgufFile)
	FfiConverterOptionalUint64INSTANCE.Write(writer, value.MemoryEstimateBytes)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.Pool)
}

type FfiDestroyerHfLoadOptionsRecord struct{}

func (_ FfiDestroyerHfLoadOptionsRecord) Destroy(value HfLoadOptionsRecord) {
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

// Kahneman-Tversky Optimization (KTO) configuration.
//
// Like DPO, KTO requires a frozen reference model (defaults to
// `core.base_model_repo`) — but the dataset schema differs:
// each row is a `(prompt, completion, desirable)` triple
// ([`UniffiRatedJsonlDataset`]), not a chosen/rejected pair.
type KtoConfigRecord struct {
	Core                   TrainCoreConfigRecord
	Lora                   LoraConfigRecord
	Beta                   float32
	LambdaD                float32
	LambdaU                float32
	ReferenceModelRepo     *string
	ReferenceModelRevision *string
}

func (r *KtoConfigRecord) Destroy() {
	FfiDestroyerTrainCoreConfigRecord{}.Destroy(r.Core)
	FfiDestroyerLoraConfigRecord{}.Destroy(r.Lora)
	FfiDestroyerFloat32{}.Destroy(r.Beta)
	FfiDestroyerFloat32{}.Destroy(r.LambdaD)
	FfiDestroyerFloat32{}.Destroy(r.LambdaU)
	FfiDestroyerOptionalString{}.Destroy(r.ReferenceModelRepo)
	FfiDestroyerOptionalString{}.Destroy(r.ReferenceModelRevision)
}

type FfiConverterKtoConfigRecord struct{}

var FfiConverterKtoConfigRecordINSTANCE = FfiConverterKtoConfigRecord{}

func (c FfiConverterKtoConfigRecord) Lift(rb RustBufferI) KtoConfigRecord {
	return LiftFromRustBuffer[KtoConfigRecord](c, rb)
}

func (c FfiConverterKtoConfigRecord) Read(reader io.Reader) KtoConfigRecord {
	return KtoConfigRecord{
		FfiConverterTrainCoreConfigRecordINSTANCE.Read(reader),
		FfiConverterLoraConfigRecordINSTANCE.Read(reader),
		FfiConverterFloat32INSTANCE.Read(reader),
		FfiConverterFloat32INSTANCE.Read(reader),
		FfiConverterFloat32INSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
	}
}

func (c FfiConverterKtoConfigRecord) Lower(value KtoConfigRecord) C.RustBuffer {
	return LowerIntoRustBuffer[KtoConfigRecord](c, value)
}

func (c FfiConverterKtoConfigRecord) LowerExternal(value KtoConfigRecord) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[KtoConfigRecord](c, value))
}

func (c FfiConverterKtoConfigRecord) Write(writer io.Writer, value KtoConfigRecord) {
	FfiConverterTrainCoreConfigRecordINSTANCE.Write(writer, value.Core)
	FfiConverterLoraConfigRecordINSTANCE.Write(writer, value.Lora)
	FfiConverterFloat32INSTANCE.Write(writer, value.Beta)
	FfiConverterFloat32INSTANCE.Write(writer, value.LambdaD)
	FfiConverterFloat32INSTANCE.Write(writer, value.LambdaU)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.ReferenceModelRepo)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.ReferenceModelRevision)
}

type FfiDestroyerKtoConfigRecord struct{}

func (_ FfiDestroyerKtoConfigRecord) Destroy(value KtoConfigRecord) {
	value.Destroy()
}

// Foreign-facing request for [`ModelClient::load_from_hf`]. Mirrors
// [`LoadFromHfRequest`] minus the `envelope_version`.
type LoadFromHfRecord struct {
	// Id under which to register the resulting model.
	ModelId string
	// Hugging Face repo slug (`org/name`).
	Repo string
	// Optional explicit memory estimate in bytes; `None` asks the
	// loader to estimate from repo metadata.
	MemoryEstimateBytes *uint64
	// Optional backend override.
	BackendHint *HfBackendHint
	// Optional GGUF file name when the backend is llama.cpp.
	GgufFile *string
	// Optional HF revision (branch / tag / commit).
	Revision *string
	// Optional bearer token for gated repos.
	HfToken *string
	// Pre-serialised JSON for any backend-specific extra options the
	// host should honor. Empty string = none.
	ExtraOptionsJson string
}

func (r *LoadFromHfRecord) Destroy() {
	FfiDestroyerString{}.Destroy(r.ModelId)
	FfiDestroyerString{}.Destroy(r.Repo)
	FfiDestroyerOptionalUint64{}.Destroy(r.MemoryEstimateBytes)
	FfiDestroyerOptionalHfBackendHint{}.Destroy(r.BackendHint)
	FfiDestroyerOptionalString{}.Destroy(r.GgufFile)
	FfiDestroyerOptionalString{}.Destroy(r.Revision)
	FfiDestroyerOptionalString{}.Destroy(r.HfToken)
	FfiDestroyerString{}.Destroy(r.ExtraOptionsJson)
}

type FfiConverterLoadFromHfRecord struct{}

var FfiConverterLoadFromHfRecordINSTANCE = FfiConverterLoadFromHfRecord{}

func (c FfiConverterLoadFromHfRecord) Lift(rb RustBufferI) LoadFromHfRecord {
	return LiftFromRustBuffer[LoadFromHfRecord](c, rb)
}

func (c FfiConverterLoadFromHfRecord) Read(reader io.Reader) LoadFromHfRecord {
	return LoadFromHfRecord{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterOptionalUint64INSTANCE.Read(reader),
		FfiConverterOptionalHfBackendHintINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
	}
}

func (c FfiConverterLoadFromHfRecord) Lower(value LoadFromHfRecord) C.RustBuffer {
	return LowerIntoRustBuffer[LoadFromHfRecord](c, value)
}

func (c FfiConverterLoadFromHfRecord) LowerExternal(value LoadFromHfRecord) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[LoadFromHfRecord](c, value))
}

func (c FfiConverterLoadFromHfRecord) Write(writer io.Writer, value LoadFromHfRecord) {
	FfiConverterStringINSTANCE.Write(writer, value.ModelId)
	FfiConverterStringINSTANCE.Write(writer, value.Repo)
	FfiConverterOptionalUint64INSTANCE.Write(writer, value.MemoryEstimateBytes)
	FfiConverterOptionalHfBackendHintINSTANCE.Write(writer, value.BackendHint)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.GgufFile)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.Revision)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.HfToken)
	FfiConverterStringINSTANCE.Write(writer, value.ExtraOptionsJson)
}

type FfiDestroyerLoadFromHfRecord struct{}

func (_ FfiDestroyerLoadFromHfRecord) Destroy(value LoadFromHfRecord) {
	value.Destroy()
}

// Foreign-facing request for [`ModelClient::load`].
//
// Mirrors [`blazen_controlplane::model_protocol::LoadRequest`] minus the
// `envelope_version` (the wrapper fills that in from
// [`MODEL_ENVELOPE_VERSION`]).
type LoadRecord struct {
	// Id under which the target model was previously registered.
	ModelId string
}

func (r *LoadRecord) Destroy() {
	FfiDestroyerString{}.Destroy(r.ModelId)
}

type FfiConverterLoadRecord struct{}

var FfiConverterLoadRecordINSTANCE = FfiConverterLoadRecord{}

func (c FfiConverterLoadRecord) Lift(rb RustBufferI) LoadRecord {
	return LiftFromRustBuffer[LoadRecord](c, rb)
}

func (c FfiConverterLoadRecord) Read(reader io.Reader) LoadRecord {
	return LoadRecord{
		FfiConverterStringINSTANCE.Read(reader),
	}
}

func (c FfiConverterLoadRecord) Lower(value LoadRecord) C.RustBuffer {
	return LowerIntoRustBuffer[LoadRecord](c, value)
}

func (c FfiConverterLoadRecord) LowerExternal(value LoadRecord) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[LoadRecord](c, value))
}

func (c FfiConverterLoadRecord) Write(writer io.Writer, value LoadRecord) {
	FfiConverterStringINSTANCE.Write(writer, value.ModelId)
}

type FfiDestroyerLoadRecord struct{}

func (_ FfiDestroyerLoadRecord) Destroy(value LoadRecord) {
	value.Destroy()
}

// Foreign-facing response from [`ModelClient::load`] and
// [`ModelClient::load_from_hf`].
//
// `LoadResponse` is empty on the wire (failures travel via the
// `Result`), but `LoadFromHfResponse` reports the chosen backend; we
// surface both through a single record with optional fields so foreign
// callers see one shape regardless of which loader they invoke.
type LoadResultRecord struct {
	// Model id that was loaded. Echoes the request's `model_id` so
	// foreign callers don't have to thread the value through their own
	// state.
	ModelId string
	// Whether the load succeeded. Always `true` on the success branch
	// of the `Result`; provided for forward-compat with future wire
	// schemas that may carry a richer status.
	Loaded bool
	// Backend the loader chose. Only populated by `load_from_hf`;
	// `None` for the plain `load` path which does not negotiate a
	// backend.
	ChosenBackend *HfBackendHint
}

func (r *LoadResultRecord) Destroy() {
	FfiDestroyerString{}.Destroy(r.ModelId)
	FfiDestroyerBool{}.Destroy(r.Loaded)
	FfiDestroyerOptionalHfBackendHint{}.Destroy(r.ChosenBackend)
}

type FfiConverterLoadResultRecord struct{}

var FfiConverterLoadResultRecordINSTANCE = FfiConverterLoadResultRecord{}

func (c FfiConverterLoadResultRecord) Lift(rb RustBufferI) LoadResultRecord {
	return LiftFromRustBuffer[LoadResultRecord](c, rb)
}

func (c FfiConverterLoadResultRecord) Read(reader io.Reader) LoadResultRecord {
	return LoadResultRecord{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterBoolINSTANCE.Read(reader),
		FfiConverterOptionalHfBackendHintINSTANCE.Read(reader),
	}
}

func (c FfiConverterLoadResultRecord) Lower(value LoadResultRecord) C.RustBuffer {
	return LowerIntoRustBuffer[LoadResultRecord](c, value)
}

func (c FfiConverterLoadResultRecord) LowerExternal(value LoadResultRecord) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[LoadResultRecord](c, value))
}

func (c FfiConverterLoadResultRecord) Write(writer io.Writer, value LoadResultRecord) {
	FfiConverterStringINSTANCE.Write(writer, value.ModelId)
	FfiConverterBoolINSTANCE.Write(writer, value.Loaded)
	FfiConverterOptionalHfBackendHintINSTANCE.Write(writer, value.ChosenBackend)
}

type FfiDestroyerLoadResultRecord struct{}

func (_ FfiDestroyerLoadResultRecord) Destroy(value LoadResultRecord) {
	value.Destroy()
}

// LoRA hyperparameters.
type LoraConfigRecord struct {
	Rank          uint32
	Alpha         float32
	Dropout       float32
	TargetModules []string
}

func (r *LoraConfigRecord) Destroy() {
	FfiDestroyerUint32{}.Destroy(r.Rank)
	FfiDestroyerFloat32{}.Destroy(r.Alpha)
	FfiDestroyerFloat32{}.Destroy(r.Dropout)
	FfiDestroyerSequenceString{}.Destroy(r.TargetModules)
}

type FfiConverterLoraConfigRecord struct{}

var FfiConverterLoraConfigRecordINSTANCE = FfiConverterLoraConfigRecord{}

func (c FfiConverterLoraConfigRecord) Lift(rb RustBufferI) LoraConfigRecord {
	return LiftFromRustBuffer[LoraConfigRecord](c, rb)
}

func (c FfiConverterLoraConfigRecord) Read(reader io.Reader) LoraConfigRecord {
	return LoraConfigRecord{
		FfiConverterUint32INSTANCE.Read(reader),
		FfiConverterFloat32INSTANCE.Read(reader),
		FfiConverterFloat32INSTANCE.Read(reader),
		FfiConverterSequenceStringINSTANCE.Read(reader),
	}
}

func (c FfiConverterLoraConfigRecord) Lower(value LoraConfigRecord) C.RustBuffer {
	return LowerIntoRustBuffer[LoraConfigRecord](c, value)
}

func (c FfiConverterLoraConfigRecord) LowerExternal(value LoraConfigRecord) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[LoraConfigRecord](c, value))
}

func (c FfiConverterLoraConfigRecord) Write(writer io.Writer, value LoraConfigRecord) {
	FfiConverterUint32INSTANCE.Write(writer, value.Rank)
	FfiConverterFloat32INSTANCE.Write(writer, value.Alpha)
	FfiConverterFloat32INSTANCE.Write(writer, value.Dropout)
	FfiConverterSequenceStringINSTANCE.Write(writer, value.TargetModules)
}

type FfiDestroyerLoraConfigRecord struct{}

func (_ FfiDestroyerLoraConfigRecord) Destroy(value LoraConfigRecord) {
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

// Foreign-facing snapshot of a single registered model.
//
// Mirrors [`ModelStatusWire`] from the model protocol. The upstream
// `adapters: Vec<AdapterStatusWire>` field is omitted in this wave —
// adapter introspection lands with the adapter RPCs in a later wave.
type ModelClientStatusRecord struct {
	// Identifier under which the model was registered.
	Id string
	// Whether the model is currently loaded into its pool.
	Loaded bool
	// Estimated memory footprint in bytes (includes any mounted adapters).
	MemoryEstimateBytes uint64
	// Pool the model is charged against.
	Pool ModelPool
}

func (r *ModelClientStatusRecord) Destroy() {
	FfiDestroyerString{}.Destroy(r.Id)
	FfiDestroyerBool{}.Destroy(r.Loaded)
	FfiDestroyerUint64{}.Destroy(r.MemoryEstimateBytes)
	FfiDestroyerModelPool{}.Destroy(r.Pool)
}

type FfiConverterModelClientStatusRecord struct{}

var FfiConverterModelClientStatusRecordINSTANCE = FfiConverterModelClientStatusRecord{}

func (c FfiConverterModelClientStatusRecord) Lift(rb RustBufferI) ModelClientStatusRecord {
	return LiftFromRustBuffer[ModelClientStatusRecord](c, rb)
}

func (c FfiConverterModelClientStatusRecord) Read(reader io.Reader) ModelClientStatusRecord {
	return ModelClientStatusRecord{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterBoolINSTANCE.Read(reader),
		FfiConverterUint64INSTANCE.Read(reader),
		FfiConverterModelPoolINSTANCE.Read(reader),
	}
}

func (c FfiConverterModelClientStatusRecord) Lower(value ModelClientStatusRecord) C.RustBuffer {
	return LowerIntoRustBuffer[ModelClientStatusRecord](c, value)
}

func (c FfiConverterModelClientStatusRecord) LowerExternal(value ModelClientStatusRecord) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[ModelClientStatusRecord](c, value))
}

func (c FfiConverterModelClientStatusRecord) Write(writer io.Writer, value ModelClientStatusRecord) {
	FfiConverterStringINSTANCE.Write(writer, value.Id)
	FfiConverterBoolINSTANCE.Write(writer, value.Loaded)
	FfiConverterUint64INSTANCE.Write(writer, value.MemoryEstimateBytes)
	FfiConverterModelPoolINSTANCE.Write(writer, value.Pool)
}

type FfiDestroyerModelClientStatusRecord struct{}

func (_ FfiDestroyerModelClientStatusRecord) Destroy(value ModelClientStatusRecord) {
	value.Destroy()
}

// A provider-agnostic chat completion request.
//
// `system`, when set, is prepended as a `Role::System` message — equivalent
// to building the message list with a leading system entry. Provided as a
// convenience because most foreign callers think of the system prompt as a
// request-level field, not a message.
type ModelRequest struct {
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

func (r *ModelRequest) Destroy() {
	FfiDestroyerSequenceChatMessage{}.Destroy(r.Messages)
	FfiDestroyerSequenceTool{}.Destroy(r.Tools)
	FfiDestroyerOptionalFloat64{}.Destroy(r.Temperature)
	FfiDestroyerOptionalUint32{}.Destroy(r.MaxTokens)
	FfiDestroyerOptionalFloat64{}.Destroy(r.TopP)
	FfiDestroyerOptionalString{}.Destroy(r.Model)
	FfiDestroyerOptionalString{}.Destroy(r.ResponseFormatJson)
	FfiDestroyerOptionalString{}.Destroy(r.System)
}

type FfiConverterModelRequest struct{}

var FfiConverterModelRequestINSTANCE = FfiConverterModelRequest{}

func (c FfiConverterModelRequest) Lift(rb RustBufferI) ModelRequest {
	return LiftFromRustBuffer[ModelRequest](c, rb)
}

func (c FfiConverterModelRequest) Read(reader io.Reader) ModelRequest {
	return ModelRequest{
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

func (c FfiConverterModelRequest) Lower(value ModelRequest) C.RustBuffer {
	return LowerIntoRustBuffer[ModelRequest](c, value)
}

func (c FfiConverterModelRequest) LowerExternal(value ModelRequest) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[ModelRequest](c, value))
}

func (c FfiConverterModelRequest) Write(writer io.Writer, value ModelRequest) {
	FfiConverterSequenceChatMessageINSTANCE.Write(writer, value.Messages)
	FfiConverterSequenceToolINSTANCE.Write(writer, value.Tools)
	FfiConverterOptionalFloat64INSTANCE.Write(writer, value.Temperature)
	FfiConverterOptionalUint32INSTANCE.Write(writer, value.MaxTokens)
	FfiConverterOptionalFloat64INSTANCE.Write(writer, value.TopP)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.Model)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.ResponseFormatJson)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.System)
}

type FfiDestroyerModelRequest struct{}

func (_ FfiDestroyerModelRequest) Destroy(value ModelRequest) {
	value.Destroy()
}

// The result of a non-streaming chat completion.
//
// `content` is the empty string when the provider returned no text (e.g.
// the model emitted only tool calls). `finish_reason` is the empty string
// when the provider didn't report one.
type ModelResponse struct {
	Content      string
	ToolCalls    []ToolCall
	FinishReason string
	Model        string
	Usage        TokenUsage
}

func (r *ModelResponse) Destroy() {
	FfiDestroyerString{}.Destroy(r.Content)
	FfiDestroyerSequenceToolCall{}.Destroy(r.ToolCalls)
	FfiDestroyerString{}.Destroy(r.FinishReason)
	FfiDestroyerString{}.Destroy(r.Model)
	FfiDestroyerTokenUsage{}.Destroy(r.Usage)
}

type FfiConverterModelResponse struct{}

var FfiConverterModelResponseINSTANCE = FfiConverterModelResponse{}

func (c FfiConverterModelResponse) Lift(rb RustBufferI) ModelResponse {
	return LiftFromRustBuffer[ModelResponse](c, rb)
}

func (c FfiConverterModelResponse) Read(reader io.Reader) ModelResponse {
	return ModelResponse{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterSequenceToolCallINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterTokenUsageINSTANCE.Read(reader),
	}
}

func (c FfiConverterModelResponse) Lower(value ModelResponse) C.RustBuffer {
	return LowerIntoRustBuffer[ModelResponse](c, value)
}

func (c FfiConverterModelResponse) LowerExternal(value ModelResponse) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[ModelResponse](c, value))
}

func (c FfiConverterModelResponse) Write(writer io.Writer, value ModelResponse) {
	FfiConverterStringINSTANCE.Write(writer, value.Content)
	FfiConverterSequenceToolCallINSTANCE.Write(writer, value.ToolCalls)
	FfiConverterStringINSTANCE.Write(writer, value.FinishReason)
	FfiConverterStringINSTANCE.Write(writer, value.Model)
	FfiConverterTokenUsageINSTANCE.Write(writer, value.Usage)
}

type FfiDestroyerModelResponse struct{}

func (_ FfiDestroyerModelResponse) Destroy(value ModelResponse) {
	value.Destroy()
}

// Per-model state snapshot returned by [`UniffiModelManager::status`].
type ModelStatusRecord struct {
	Id                  string
	Loaded              bool
	MemoryEstimateBytes uint64
	// Pool label (`"cpu"` or `"gpu:N"`).
	Pool     string
	Adapters []AdapterStatusRecord
}

func (r *ModelStatusRecord) Destroy() {
	FfiDestroyerString{}.Destroy(r.Id)
	FfiDestroyerBool{}.Destroy(r.Loaded)
	FfiDestroyerUint64{}.Destroy(r.MemoryEstimateBytes)
	FfiDestroyerString{}.Destroy(r.Pool)
	FfiDestroyerSequenceAdapterStatusRecord{}.Destroy(r.Adapters)
}

type FfiConverterModelStatusRecord struct{}

var FfiConverterModelStatusRecordINSTANCE = FfiConverterModelStatusRecord{}

func (c FfiConverterModelStatusRecord) Lift(rb RustBufferI) ModelStatusRecord {
	return LiftFromRustBuffer[ModelStatusRecord](c, rb)
}

func (c FfiConverterModelStatusRecord) Read(reader io.Reader) ModelStatusRecord {
	return ModelStatusRecord{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterBoolINSTANCE.Read(reader),
		FfiConverterUint64INSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterSequenceAdapterStatusRecordINSTANCE.Read(reader),
	}
}

func (c FfiConverterModelStatusRecord) Lower(value ModelStatusRecord) C.RustBuffer {
	return LowerIntoRustBuffer[ModelStatusRecord](c, value)
}

func (c FfiConverterModelStatusRecord) LowerExternal(value ModelStatusRecord) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[ModelStatusRecord](c, value))
}

func (c FfiConverterModelStatusRecord) Write(writer io.Writer, value ModelStatusRecord) {
	FfiConverterStringINSTANCE.Write(writer, value.Id)
	FfiConverterBoolINSTANCE.Write(writer, value.Loaded)
	FfiConverterUint64INSTANCE.Write(writer, value.MemoryEstimateBytes)
	FfiConverterStringINSTANCE.Write(writer, value.Pool)
	FfiConverterSequenceAdapterStatusRecordINSTANCE.Write(writer, value.Adapters)
}

type FfiDestroyerModelStatusRecord struct{}

func (_ FfiDestroyerModelStatusRecord) Destroy(value ModelStatusRecord) {
	value.Destroy()
}

// One emission from a streaming music backend.
//
// `samples` is 32-bit float PCM in `[-1.0, 1.0]` at the backend's expected
// output sample rate (the same `sample_rate` field on the
// [`MusicResult`] returned by the non-streaming
// `generate_music` / `generate_sfx` calls).
//
// `is_final` is `true` for the final chunk of a generation call;
// implementations should treat it as a UI hint rather than the
// authoritative completion signal — the sink's `on_done` callback is the
// canonical end-of-stream marker.
//
// `latency_seconds`, when present, is the measured latency from the
// stream's call-start to the moment this chunk was produced — handy for
// surfacing first-token-latency metrics through the binding.
type MusicChunk struct {
	// 32-bit float PCM samples in `[-1, 1]` at the backend's sample rate.
	Samples []float32
	// `true` on the final emitted chunk; otherwise `false`.
	IsFinal bool
	// Optional per-chunk latency from call-start in seconds.
	LatencySeconds *float32
}

func (r *MusicChunk) Destroy() {
	FfiDestroyerSequenceFloat32{}.Destroy(r.Samples)
	FfiDestroyerBool{}.Destroy(r.IsFinal)
	FfiDestroyerOptionalFloat32{}.Destroy(r.LatencySeconds)
}

type FfiConverterMusicChunk struct{}

var FfiConverterMusicChunkINSTANCE = FfiConverterMusicChunk{}

func (c FfiConverterMusicChunk) Lift(rb RustBufferI) MusicChunk {
	return LiftFromRustBuffer[MusicChunk](c, rb)
}

func (c FfiConverterMusicChunk) Read(reader io.Reader) MusicChunk {
	return MusicChunk{
		FfiConverterSequenceFloat32INSTANCE.Read(reader),
		FfiConverterBoolINSTANCE.Read(reader),
		FfiConverterOptionalFloat32INSTANCE.Read(reader),
	}
}

func (c FfiConverterMusicChunk) Lower(value MusicChunk) C.RustBuffer {
	return LowerIntoRustBuffer[MusicChunk](c, value)
}

func (c FfiConverterMusicChunk) LowerExternal(value MusicChunk) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[MusicChunk](c, value))
}

func (c FfiConverterMusicChunk) Write(writer io.Writer, value MusicChunk) {
	FfiConverterSequenceFloat32INSTANCE.Write(writer, value.Samples)
	FfiConverterBoolINSTANCE.Write(writer, value.IsFinal)
	FfiConverterOptionalFloat32INSTANCE.Write(writer, value.LatencySeconds)
}

type FfiDestroyerMusicChunk struct{}

func (_ FfiDestroyerMusicChunk) Destroy(value MusicChunk) {
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

// A fully-rendered music / SFX result.
//
// `bytes` carries the encoded audio (typically a WAV container for the
// native backends; whatever the cloud provider returned for fal.ai). The
// non-empty `url` field signals a URL-only response (e.g. fal.ai returning
// a CDN link without inlining bytes); `bytes` will be empty in that case.
// Callers should pick whichever payload is present.
type MusicResult struct {
	// Encoded audio bytes. Empty when the upstream provider only returned
	// a URL.
	Bytes []byte
	// IANA MIME type of `bytes` (e.g. `"audio/wav"`, `"audio/mpeg"`).
	MimeType string
	// Sample rate in Hz. Zero when the upstream provider didn't report
	// one.
	SampleRate uint32
	// Channel count (1 = mono, 2 = stereo). Zero when the upstream
	// provider didn't report it.
	Channels uint32
	// Duration of the clip in seconds. Zero when the upstream provider
	// didn't report a duration.
	DurationSeconds float32
	// URL of the audio asset when the upstream provider only returned a
	// link. Empty string for inline-bytes results.
	Url string
}

func (r *MusicResult) Destroy() {
	FfiDestroyerBytes{}.Destroy(r.Bytes)
	FfiDestroyerString{}.Destroy(r.MimeType)
	FfiDestroyerUint32{}.Destroy(r.SampleRate)
	FfiDestroyerUint32{}.Destroy(r.Channels)
	FfiDestroyerFloat32{}.Destroy(r.DurationSeconds)
	FfiDestroyerString{}.Destroy(r.Url)
}

type FfiConverterMusicResult struct{}

var FfiConverterMusicResultINSTANCE = FfiConverterMusicResult{}

func (c FfiConverterMusicResult) Lift(rb RustBufferI) MusicResult {
	return LiftFromRustBuffer[MusicResult](c, rb)
}

func (c FfiConverterMusicResult) Read(reader io.Reader) MusicResult {
	return MusicResult{
		FfiConverterBytesINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterUint32INSTANCE.Read(reader),
		FfiConverterUint32INSTANCE.Read(reader),
		FfiConverterFloat32INSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
	}
}

func (c FfiConverterMusicResult) Lower(value MusicResult) C.RustBuffer {
	return LowerIntoRustBuffer[MusicResult](c, value)
}

func (c FfiConverterMusicResult) LowerExternal(value MusicResult) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[MusicResult](c, value))
}

func (c FfiConverterMusicResult) Write(writer io.Writer, value MusicResult) {
	FfiConverterBytesINSTANCE.Write(writer, value.Bytes)
	FfiConverterStringINSTANCE.Write(writer, value.MimeType)
	FfiConverterUint32INSTANCE.Write(writer, value.SampleRate)
	FfiConverterUint32INSTANCE.Write(writer, value.Channels)
	FfiConverterFloat32INSTANCE.Write(writer, value.DurationSeconds)
	FfiConverterStringINSTANCE.Write(writer, value.Url)
}

type FfiDestroyerMusicResult struct{}

func (_ FfiDestroyerMusicResult) Destroy(value MusicResult) {
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

// AdamW optimizer hyperparameters.
type OptimConfigRecord struct {
	LearningRate float64
	Beta1        float64
	Beta2        float64
	Epsilon      float64
	WeightDecay  float64
	GradientClip *float32
}

func (r *OptimConfigRecord) Destroy() {
	FfiDestroyerFloat64{}.Destroy(r.LearningRate)
	FfiDestroyerFloat64{}.Destroy(r.Beta1)
	FfiDestroyerFloat64{}.Destroy(r.Beta2)
	FfiDestroyerFloat64{}.Destroy(r.Epsilon)
	FfiDestroyerFloat64{}.Destroy(r.WeightDecay)
	FfiDestroyerOptionalFloat32{}.Destroy(r.GradientClip)
}

type FfiConverterOptimConfigRecord struct{}

var FfiConverterOptimConfigRecordINSTANCE = FfiConverterOptimConfigRecord{}

func (c FfiConverterOptimConfigRecord) Lift(rb RustBufferI) OptimConfigRecord {
	return LiftFromRustBuffer[OptimConfigRecord](c, rb)
}

func (c FfiConverterOptimConfigRecord) Read(reader io.Reader) OptimConfigRecord {
	return OptimConfigRecord{
		FfiConverterFloat64INSTANCE.Read(reader),
		FfiConverterFloat64INSTANCE.Read(reader),
		FfiConverterFloat64INSTANCE.Read(reader),
		FfiConverterFloat64INSTANCE.Read(reader),
		FfiConverterFloat64INSTANCE.Read(reader),
		FfiConverterOptionalFloat32INSTANCE.Read(reader),
	}
}

func (c FfiConverterOptimConfigRecord) Lower(value OptimConfigRecord) C.RustBuffer {
	return LowerIntoRustBuffer[OptimConfigRecord](c, value)
}

func (c FfiConverterOptimConfigRecord) LowerExternal(value OptimConfigRecord) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[OptimConfigRecord](c, value))
}

func (c FfiConverterOptimConfigRecord) Write(writer io.Writer, value OptimConfigRecord) {
	FfiConverterFloat64INSTANCE.Write(writer, value.LearningRate)
	FfiConverterFloat64INSTANCE.Write(writer, value.Beta1)
	FfiConverterFloat64INSTANCE.Write(writer, value.Beta2)
	FfiConverterFloat64INSTANCE.Write(writer, value.Epsilon)
	FfiConverterFloat64INSTANCE.Write(writer, value.WeightDecay)
	FfiConverterOptionalFloat32INSTANCE.Write(writer, value.GradientClip)
}

type FfiDestroyerOptimConfigRecord struct{}

func (_ FfiDestroyerOptimConfigRecord) Destroy(value OptimConfigRecord) {
	value.Destroy()
}

// Odds Ratio Preference Optimization (ORPO) configuration.
//
// Reference-free — combines an SFT loss on chosen responses with an
// odds-ratio loss term weighted by `lambda`.
type OrpoConfigRecord struct {
	Core   TrainCoreConfigRecord
	Lora   LoraConfigRecord
	Lambda float32
}

func (r *OrpoConfigRecord) Destroy() {
	FfiDestroyerTrainCoreConfigRecord{}.Destroy(r.Core)
	FfiDestroyerLoraConfigRecord{}.Destroy(r.Lora)
	FfiDestroyerFloat32{}.Destroy(r.Lambda)
}

type FfiConverterOrpoConfigRecord struct{}

var FfiConverterOrpoConfigRecordINSTANCE = FfiConverterOrpoConfigRecord{}

func (c FfiConverterOrpoConfigRecord) Lift(rb RustBufferI) OrpoConfigRecord {
	return LiftFromRustBuffer[OrpoConfigRecord](c, rb)
}

func (c FfiConverterOrpoConfigRecord) Read(reader io.Reader) OrpoConfigRecord {
	return OrpoConfigRecord{
		FfiConverterTrainCoreConfigRecordINSTANCE.Read(reader),
		FfiConverterLoraConfigRecordINSTANCE.Read(reader),
		FfiConverterFloat32INSTANCE.Read(reader),
	}
}

func (c FfiConverterOrpoConfigRecord) Lower(value OrpoConfigRecord) C.RustBuffer {
	return LowerIntoRustBuffer[OrpoConfigRecord](c, value)
}

func (c FfiConverterOrpoConfigRecord) LowerExternal(value OrpoConfigRecord) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[OrpoConfigRecord](c, value))
}

func (c FfiConverterOrpoConfigRecord) Write(writer io.Writer, value OrpoConfigRecord) {
	FfiConverterTrainCoreConfigRecordINSTANCE.Write(writer, value.Core)
	FfiConverterLoraConfigRecordINSTANCE.Write(writer, value.Lora)
	FfiConverterFloat32INSTANCE.Write(writer, value.Lambda)
}

type FfiDestroyerOrpoConfigRecord struct{}

func (_ FfiDestroyerOrpoConfigRecord) Destroy(value OrpoConfigRecord) {
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

// Per-pool budget snapshot returned by [`UniffiModelManager::pools`].
type PoolStatusRecord struct {
	// Pool label (`"cpu"` or `"gpu:N"`).
	Pool        string
	BudgetBytes uint64
}

func (r *PoolStatusRecord) Destroy() {
	FfiDestroyerString{}.Destroy(r.Pool)
	FfiDestroyerUint64{}.Destroy(r.BudgetBytes)
}

type FfiConverterPoolStatusRecord struct{}

var FfiConverterPoolStatusRecordINSTANCE = FfiConverterPoolStatusRecord{}

func (c FfiConverterPoolStatusRecord) Lift(rb RustBufferI) PoolStatusRecord {
	return LiftFromRustBuffer[PoolStatusRecord](c, rb)
}

func (c FfiConverterPoolStatusRecord) Read(reader io.Reader) PoolStatusRecord {
	return PoolStatusRecord{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterUint64INSTANCE.Read(reader),
	}
}

func (c FfiConverterPoolStatusRecord) Lower(value PoolStatusRecord) C.RustBuffer {
	return LowerIntoRustBuffer[PoolStatusRecord](c, value)
}

func (c FfiConverterPoolStatusRecord) LowerExternal(value PoolStatusRecord) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[PoolStatusRecord](c, value))
}

func (c FfiConverterPoolStatusRecord) Write(writer io.Writer, value PoolStatusRecord) {
	FfiConverterStringINSTANCE.Write(writer, value.Pool)
	FfiConverterUint64INSTANCE.Write(writer, value.BudgetBytes)
}

type FfiDestroyerPoolStatusRecord struct{}

func (_ FfiDestroyerPoolStatusRecord) Destroy(value PoolStatusRecord) {
	value.Destroy()
}

// Completion-role defaults: system prompt, default tools, default
// `response_format`. Hooks (`before_model`) deferred to Phase C.
type ProviderDefaults struct {
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

func (r *ProviderDefaults) Destroy() {
	FfiDestroyerOptionalBaseProviderDefaults{}.Destroy(r.Base)
	FfiDestroyerOptionalString{}.Destroy(r.SystemPrompt)
	FfiDestroyerOptionalString{}.Destroy(r.ToolsJson)
	FfiDestroyerOptionalString{}.Destroy(r.ResponseFormatJson)
}

type FfiConverterProviderDefaults struct{}

var FfiConverterProviderDefaultsINSTANCE = FfiConverterProviderDefaults{}

func (c FfiConverterProviderDefaults) Lift(rb RustBufferI) ProviderDefaults {
	return LiftFromRustBuffer[ProviderDefaults](c, rb)
}

func (c FfiConverterProviderDefaults) Read(reader io.Reader) ProviderDefaults {
	return ProviderDefaults{
		FfiConverterOptionalBaseProviderDefaultsINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
	}
}

func (c FfiConverterProviderDefaults) Lower(value ProviderDefaults) C.RustBuffer {
	return LowerIntoRustBuffer[ProviderDefaults](c, value)
}

func (c FfiConverterProviderDefaults) LowerExternal(value ProviderDefaults) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[ProviderDefaults](c, value))
}

func (c FfiConverterProviderDefaults) Write(writer io.Writer, value ProviderDefaults) {
	FfiConverterOptionalBaseProviderDefaultsINSTANCE.Write(writer, value.Base)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.SystemPrompt)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.ToolsJson)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.ResponseFormatJson)
}

type FfiDestroyerProviderDefaults struct{}

func (_ FfiDestroyerProviderDefaults) Destroy(value ProviderDefaults) {
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

// Learning-rate scheduler configuration.
type SchedulerConfigRecord struct {
	Kind        SchedulerKindEnum
	WarmupSteps uint32
}

func (r *SchedulerConfigRecord) Destroy() {
	FfiDestroyerSchedulerKindEnum{}.Destroy(r.Kind)
	FfiDestroyerUint32{}.Destroy(r.WarmupSteps)
}

type FfiConverterSchedulerConfigRecord struct{}

var FfiConverterSchedulerConfigRecordINSTANCE = FfiConverterSchedulerConfigRecord{}

func (c FfiConverterSchedulerConfigRecord) Lift(rb RustBufferI) SchedulerConfigRecord {
	return LiftFromRustBuffer[SchedulerConfigRecord](c, rb)
}

func (c FfiConverterSchedulerConfigRecord) Read(reader io.Reader) SchedulerConfigRecord {
	return SchedulerConfigRecord{
		FfiConverterSchedulerKindEnumINSTANCE.Read(reader),
		FfiConverterUint32INSTANCE.Read(reader),
	}
}

func (c FfiConverterSchedulerConfigRecord) Lower(value SchedulerConfigRecord) C.RustBuffer {
	return LowerIntoRustBuffer[SchedulerConfigRecord](c, value)
}

func (c FfiConverterSchedulerConfigRecord) LowerExternal(value SchedulerConfigRecord) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[SchedulerConfigRecord](c, value))
}

func (c FfiConverterSchedulerConfigRecord) Write(writer io.Writer, value SchedulerConfigRecord) {
	FfiConverterSchedulerKindEnumINSTANCE.Write(writer, value.Kind)
	FfiConverterUint32INSTANCE.Write(writer, value.WarmupSteps)
}

type FfiDestroyerSchedulerConfigRecord struct{}

func (_ FfiDestroyerSchedulerConfigRecord) Destroy(value SchedulerConfigRecord) {
	value.Destroy()
}

// Simple Preference Optimization (`SimPO`) configuration.
//
// Reference-free, length-normalized. Defaults follow TRL `main`
// (`beta = 2.0`, `gamma = 1.0`).
type SimpoConfigRecord struct {
	Core  TrainCoreConfigRecord
	Lora  LoraConfigRecord
	Beta  float32
	Gamma float32
}

func (r *SimpoConfigRecord) Destroy() {
	FfiDestroyerTrainCoreConfigRecord{}.Destroy(r.Core)
	FfiDestroyerLoraConfigRecord{}.Destroy(r.Lora)
	FfiDestroyerFloat32{}.Destroy(r.Beta)
	FfiDestroyerFloat32{}.Destroy(r.Gamma)
}

type FfiConverterSimpoConfigRecord struct{}

var FfiConverterSimpoConfigRecordINSTANCE = FfiConverterSimpoConfigRecord{}

func (c FfiConverterSimpoConfigRecord) Lift(rb RustBufferI) SimpoConfigRecord {
	return LiftFromRustBuffer[SimpoConfigRecord](c, rb)
}

func (c FfiConverterSimpoConfigRecord) Read(reader io.Reader) SimpoConfigRecord {
	return SimpoConfigRecord{
		FfiConverterTrainCoreConfigRecordINSTANCE.Read(reader),
		FfiConverterLoraConfigRecordINSTANCE.Read(reader),
		FfiConverterFloat32INSTANCE.Read(reader),
		FfiConverterFloat32INSTANCE.Read(reader),
	}
}

func (c FfiConverterSimpoConfigRecord) Lower(value SimpoConfigRecord) C.RustBuffer {
	return LowerIntoRustBuffer[SimpoConfigRecord](c, value)
}

func (c FfiConverterSimpoConfigRecord) LowerExternal(value SimpoConfigRecord) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[SimpoConfigRecord](c, value))
}

func (c FfiConverterSimpoConfigRecord) Write(writer io.Writer, value SimpoConfigRecord) {
	FfiConverterTrainCoreConfigRecordINSTANCE.Write(writer, value.Core)
	FfiConverterLoraConfigRecordINSTANCE.Write(writer, value.Lora)
	FfiConverterFloat32INSTANCE.Write(writer, value.Beta)
	FfiConverterFloat32INSTANCE.Write(writer, value.Gamma)
}

type FfiDestroyerSimpoConfigRecord struct{}

func (_ FfiDestroyerSimpoConfigRecord) Destroy(value SimpoConfigRecord) {
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

// Foreign-facing response from
// [`ModelClient::status`]. Mirrors
// [`blazen_controlplane::model_protocol::StatusResponse`] but filters
// to a single model when the caller scopes the query with a
// `model_id`.
type StatusRecord struct {
	// Snapshot of each registered model (or just the requested one,
	// when `status(Some(id))` was called).
	Models []ModelClientStatusRecord
}

func (r *StatusRecord) Destroy() {
	FfiDestroyerSequenceModelClientStatusRecord{}.Destroy(r.Models)
}

type FfiConverterStatusRecord struct{}

var FfiConverterStatusRecordINSTANCE = FfiConverterStatusRecord{}

func (c FfiConverterStatusRecord) Lift(rb RustBufferI) StatusRecord {
	return LiftFromRustBuffer[StatusRecord](c, rb)
}

func (c FfiConverterStatusRecord) Read(reader io.Reader) StatusRecord {
	return StatusRecord{
		FfiConverterSequenceModelClientStatusRecordINSTANCE.Read(reader),
	}
}

func (c FfiConverterStatusRecord) Lower(value StatusRecord) C.RustBuffer {
	return LowerIntoRustBuffer[StatusRecord](c, value)
}

func (c FfiConverterStatusRecord) LowerExternal(value StatusRecord) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[StatusRecord](c, value))
}

func (c FfiConverterStatusRecord) Write(writer io.Writer, value StatusRecord) {
	FfiConverterSequenceModelClientStatusRecordINSTANCE.Write(writer, value.Models)
}

type FfiDestroyerStatusRecord struct{}

func (_ FfiDestroyerStatusRecord) Destroy(value StatusRecord) {
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

// A registered target speaker that a [`VcModel`] can render source audio
// into.
//
// Mirrors [`blazen_llm::TargetVoice`] (when the `audio-vc` feature is on)
// 1:1 across the FFI boundary so foreign code sees a stable record shape
// regardless of whether the underlying engine is the native RVC backend
// or a cloud-side provider added later.
type TargetVoice struct {
	// Backend-scoped identifier passed back to
	// [`VcModel::convert_voice`] / [`VcModel::register_target_voice`].
	Id string
	// Optional human-readable display name. `None` when the backend did
	// not record one.
	Label *string
	// Native sample rate (Hz) the backend renders this voice at.
	SampleRateHz uint32
}

func (r *TargetVoice) Destroy() {
	FfiDestroyerString{}.Destroy(r.Id)
	FfiDestroyerOptionalString{}.Destroy(r.Label)
	FfiDestroyerUint32{}.Destroy(r.SampleRateHz)
}

type FfiConverterTargetVoice struct{}

var FfiConverterTargetVoiceINSTANCE = FfiConverterTargetVoice{}

func (c FfiConverterTargetVoice) Lift(rb RustBufferI) TargetVoice {
	return LiftFromRustBuffer[TargetVoice](c, rb)
}

func (c FfiConverterTargetVoice) Read(reader io.Reader) TargetVoice {
	return TargetVoice{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterUint32INSTANCE.Read(reader),
	}
}

func (c FfiConverterTargetVoice) Lower(value TargetVoice) C.RustBuffer {
	return LowerIntoRustBuffer[TargetVoice](c, value)
}

func (c FfiConverterTargetVoice) LowerExternal(value TargetVoice) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[TargetVoice](c, value))
}

func (c FfiConverterTargetVoice) Write(writer io.Writer, value TargetVoice) {
	FfiConverterStringINSTANCE.Write(writer, value.Id)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.Label)
	FfiConverterUint32INSTANCE.Write(writer, value.SampleRateHz)
}

type FfiDestroyerTargetVoice struct{}

func (_ FfiDestroyerTargetVoice) Destroy(value TargetVoice) {
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

// Full configuration for one training run.
type TrainConfigRecord struct {
	BaseModelRepo             string
	OutputDir                 string
	Lora                      LoraConfigRecord
	Optim                     OptimConfigRecord
	Scheduler                 SchedulerConfigRecord
	MaxSteps                  uint32
	BatchSize                 uint32
	GradientAccumulationSteps uint32
	MaxSeqLen                 uint32
	EvalSteps                 *uint32
	SaveSteps                 *uint32
	Seed                      uint64
	MixedPrecision            MixedPrecisionEnum
	Device                    *string
}

func (r *TrainConfigRecord) Destroy() {
	FfiDestroyerString{}.Destroy(r.BaseModelRepo)
	FfiDestroyerString{}.Destroy(r.OutputDir)
	FfiDestroyerLoraConfigRecord{}.Destroy(r.Lora)
	FfiDestroyerOptimConfigRecord{}.Destroy(r.Optim)
	FfiDestroyerSchedulerConfigRecord{}.Destroy(r.Scheduler)
	FfiDestroyerUint32{}.Destroy(r.MaxSteps)
	FfiDestroyerUint32{}.Destroy(r.BatchSize)
	FfiDestroyerUint32{}.Destroy(r.GradientAccumulationSteps)
	FfiDestroyerUint32{}.Destroy(r.MaxSeqLen)
	FfiDestroyerOptionalUint32{}.Destroy(r.EvalSteps)
	FfiDestroyerOptionalUint32{}.Destroy(r.SaveSteps)
	FfiDestroyerUint64{}.Destroy(r.Seed)
	FfiDestroyerMixedPrecisionEnum{}.Destroy(r.MixedPrecision)
	FfiDestroyerOptionalString{}.Destroy(r.Device)
}

type FfiConverterTrainConfigRecord struct{}

var FfiConverterTrainConfigRecordINSTANCE = FfiConverterTrainConfigRecord{}

func (c FfiConverterTrainConfigRecord) Lift(rb RustBufferI) TrainConfigRecord {
	return LiftFromRustBuffer[TrainConfigRecord](c, rb)
}

func (c FfiConverterTrainConfigRecord) Read(reader io.Reader) TrainConfigRecord {
	return TrainConfigRecord{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterLoraConfigRecordINSTANCE.Read(reader),
		FfiConverterOptimConfigRecordINSTANCE.Read(reader),
		FfiConverterSchedulerConfigRecordINSTANCE.Read(reader),
		FfiConverterUint32INSTANCE.Read(reader),
		FfiConverterUint32INSTANCE.Read(reader),
		FfiConverterUint32INSTANCE.Read(reader),
		FfiConverterUint32INSTANCE.Read(reader),
		FfiConverterOptionalUint32INSTANCE.Read(reader),
		FfiConverterOptionalUint32INSTANCE.Read(reader),
		FfiConverterUint64INSTANCE.Read(reader),
		FfiConverterMixedPrecisionEnumINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
	}
}

func (c FfiConverterTrainConfigRecord) Lower(value TrainConfigRecord) C.RustBuffer {
	return LowerIntoRustBuffer[TrainConfigRecord](c, value)
}

func (c FfiConverterTrainConfigRecord) LowerExternal(value TrainConfigRecord) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[TrainConfigRecord](c, value))
}

func (c FfiConverterTrainConfigRecord) Write(writer io.Writer, value TrainConfigRecord) {
	FfiConverterStringINSTANCE.Write(writer, value.BaseModelRepo)
	FfiConverterStringINSTANCE.Write(writer, value.OutputDir)
	FfiConverterLoraConfigRecordINSTANCE.Write(writer, value.Lora)
	FfiConverterOptimConfigRecordINSTANCE.Write(writer, value.Optim)
	FfiConverterSchedulerConfigRecordINSTANCE.Write(writer, value.Scheduler)
	FfiConverterUint32INSTANCE.Write(writer, value.MaxSteps)
	FfiConverterUint32INSTANCE.Write(writer, value.BatchSize)
	FfiConverterUint32INSTANCE.Write(writer, value.GradientAccumulationSteps)
	FfiConverterUint32INSTANCE.Write(writer, value.MaxSeqLen)
	FfiConverterOptionalUint32INSTANCE.Write(writer, value.EvalSteps)
	FfiConverterOptionalUint32INSTANCE.Write(writer, value.SaveSteps)
	FfiConverterUint64INSTANCE.Write(writer, value.Seed)
	FfiConverterMixedPrecisionEnumINSTANCE.Write(writer, value.MixedPrecision)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.Device)
}

type FfiDestroyerTrainConfigRecord struct{}

func (_ FfiDestroyerTrainConfigRecord) Destroy(value TrainConfigRecord) {
	value.Destroy()
}

// Shared training hyperparameters used by DPO / ORPO / SimPO / KTO /
// full fine-tune. Mirrors [`TrainCoreConfig`] (`TrainConfig` minus the
// PEFT-specific [`LoraConfigRecord`]).
type TrainCoreConfigRecord struct {
	BaseModelRepo             string
	BaseModelRevision         *string
	OutputDir                 string
	MaxSteps                  uint32
	BatchSize                 uint32
	GradientAccumulationSteps uint32
	MaxSeqLen                 uint32
	EvalSteps                 *uint32
	SaveSteps                 *uint32
	Seed                      uint64
	MixedPrecision            MixedPrecisionEnum
	Device                    *string
	Optim                     OptimConfigRecord
	Scheduler                 SchedulerConfigRecord
}

func (r *TrainCoreConfigRecord) Destroy() {
	FfiDestroyerString{}.Destroy(r.BaseModelRepo)
	FfiDestroyerOptionalString{}.Destroy(r.BaseModelRevision)
	FfiDestroyerString{}.Destroy(r.OutputDir)
	FfiDestroyerUint32{}.Destroy(r.MaxSteps)
	FfiDestroyerUint32{}.Destroy(r.BatchSize)
	FfiDestroyerUint32{}.Destroy(r.GradientAccumulationSteps)
	FfiDestroyerUint32{}.Destroy(r.MaxSeqLen)
	FfiDestroyerOptionalUint32{}.Destroy(r.EvalSteps)
	FfiDestroyerOptionalUint32{}.Destroy(r.SaveSteps)
	FfiDestroyerUint64{}.Destroy(r.Seed)
	FfiDestroyerMixedPrecisionEnum{}.Destroy(r.MixedPrecision)
	FfiDestroyerOptionalString{}.Destroy(r.Device)
	FfiDestroyerOptimConfigRecord{}.Destroy(r.Optim)
	FfiDestroyerSchedulerConfigRecord{}.Destroy(r.Scheduler)
}

type FfiConverterTrainCoreConfigRecord struct{}

var FfiConverterTrainCoreConfigRecordINSTANCE = FfiConverterTrainCoreConfigRecord{}

func (c FfiConverterTrainCoreConfigRecord) Lift(rb RustBufferI) TrainCoreConfigRecord {
	return LiftFromRustBuffer[TrainCoreConfigRecord](c, rb)
}

func (c FfiConverterTrainCoreConfigRecord) Read(reader io.Reader) TrainCoreConfigRecord {
	return TrainCoreConfigRecord{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterUint32INSTANCE.Read(reader),
		FfiConverterUint32INSTANCE.Read(reader),
		FfiConverterUint32INSTANCE.Read(reader),
		FfiConverterUint32INSTANCE.Read(reader),
		FfiConverterOptionalUint32INSTANCE.Read(reader),
		FfiConverterOptionalUint32INSTANCE.Read(reader),
		FfiConverterUint64INSTANCE.Read(reader),
		FfiConverterMixedPrecisionEnumINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterOptimConfigRecordINSTANCE.Read(reader),
		FfiConverterSchedulerConfigRecordINSTANCE.Read(reader),
	}
}

func (c FfiConverterTrainCoreConfigRecord) Lower(value TrainCoreConfigRecord) C.RustBuffer {
	return LowerIntoRustBuffer[TrainCoreConfigRecord](c, value)
}

func (c FfiConverterTrainCoreConfigRecord) LowerExternal(value TrainCoreConfigRecord) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[TrainCoreConfigRecord](c, value))
}

func (c FfiConverterTrainCoreConfigRecord) Write(writer io.Writer, value TrainCoreConfigRecord) {
	FfiConverterStringINSTANCE.Write(writer, value.BaseModelRepo)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.BaseModelRevision)
	FfiConverterStringINSTANCE.Write(writer, value.OutputDir)
	FfiConverterUint32INSTANCE.Write(writer, value.MaxSteps)
	FfiConverterUint32INSTANCE.Write(writer, value.BatchSize)
	FfiConverterUint32INSTANCE.Write(writer, value.GradientAccumulationSteps)
	FfiConverterUint32INSTANCE.Write(writer, value.MaxSeqLen)
	FfiConverterOptionalUint32INSTANCE.Write(writer, value.EvalSteps)
	FfiConverterOptionalUint32INSTANCE.Write(writer, value.SaveSteps)
	FfiConverterUint64INSTANCE.Write(writer, value.Seed)
	FfiConverterMixedPrecisionEnumINSTANCE.Write(writer, value.MixedPrecision)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.Device)
	FfiConverterOptimConfigRecordINSTANCE.Write(writer, value.Optim)
	FfiConverterSchedulerConfigRecordINSTANCE.Write(writer, value.Scheduler)
}

type FfiDestroyerTrainCoreConfigRecord struct{}

func (_ FfiDestroyerTrainCoreConfigRecord) Destroy(value TrainCoreConfigRecord) {
	value.Destroy()
}

// On-disk descriptor returned by [`UniffiModelManager::train_lora`].
type TrainedAdapterRecord struct {
	AdapterDir string
	FinalLoss  float32
	TotalSteps uint64
}

func (r *TrainedAdapterRecord) Destroy() {
	FfiDestroyerString{}.Destroy(r.AdapterDir)
	FfiDestroyerFloat32{}.Destroy(r.FinalLoss)
	FfiDestroyerUint64{}.Destroy(r.TotalSteps)
}

type FfiConverterTrainedAdapterRecord struct{}

var FfiConverterTrainedAdapterRecordINSTANCE = FfiConverterTrainedAdapterRecord{}

func (c FfiConverterTrainedAdapterRecord) Lift(rb RustBufferI) TrainedAdapterRecord {
	return LiftFromRustBuffer[TrainedAdapterRecord](c, rb)
}

func (c FfiConverterTrainedAdapterRecord) Read(reader io.Reader) TrainedAdapterRecord {
	return TrainedAdapterRecord{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterFloat32INSTANCE.Read(reader),
		FfiConverterUint64INSTANCE.Read(reader),
	}
}

func (c FfiConverterTrainedAdapterRecord) Lower(value TrainedAdapterRecord) C.RustBuffer {
	return LowerIntoRustBuffer[TrainedAdapterRecord](c, value)
}

func (c FfiConverterTrainedAdapterRecord) LowerExternal(value TrainedAdapterRecord) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[TrainedAdapterRecord](c, value))
}

func (c FfiConverterTrainedAdapterRecord) Write(writer io.Writer, value TrainedAdapterRecord) {
	FfiConverterStringINSTANCE.Write(writer, value.AdapterDir)
	FfiConverterFloat32INSTANCE.Write(writer, value.FinalLoss)
	FfiConverterUint64INSTANCE.Write(writer, value.TotalSteps)
}

type FfiDestroyerTrainedAdapterRecord struct{}

func (_ FfiDestroyerTrainedAdapterRecord) Destroy(value TrainedAdapterRecord) {
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

// One emission from a streaming voice-conversion call.
//
// `samples` is 32-bit float PCM in `[-1.0, 1.0]` at the target voice's
// native sample rate (see [`TargetVoice::sample_rate_hz`]).
//
// `is_final` is purely an advisory hint — the sink's `on_done` callback
// is the canonical end-of-stream signal, matching the contract used by
// [`crate::compute_music::MusicChunk`].
//
// `latency_seconds`, when present, is the measured latency from the
// stream's call-start to the moment this chunk was produced — handy for
// surfacing first-token-latency metrics through the binding.
type VcChunk struct {
	// 32-bit float PCM samples in `[-1, 1]` at the voice's native sample
	// rate.
	Samples []float32
	// `true` on the final emitted chunk; otherwise `false`. Always
	// `false` for the RVC backend today (end-of-stream is signalled by
	// the sink's `on_done` callback).
	IsFinal bool
	// Optional per-chunk latency from call-start in seconds.
	LatencySeconds *float32
}

func (r *VcChunk) Destroy() {
	FfiDestroyerSequenceFloat32{}.Destroy(r.Samples)
	FfiDestroyerBool{}.Destroy(r.IsFinal)
	FfiDestroyerOptionalFloat32{}.Destroy(r.LatencySeconds)
}

type FfiConverterVcChunk struct{}

var FfiConverterVcChunkINSTANCE = FfiConverterVcChunk{}

func (c FfiConverterVcChunk) Lift(rb RustBufferI) VcChunk {
	return LiftFromRustBuffer[VcChunk](c, rb)
}

func (c FfiConverterVcChunk) Read(reader io.Reader) VcChunk {
	return VcChunk{
		FfiConverterSequenceFloat32INSTANCE.Read(reader),
		FfiConverterBoolINSTANCE.Read(reader),
		FfiConverterOptionalFloat32INSTANCE.Read(reader),
	}
}

func (c FfiConverterVcChunk) Lower(value VcChunk) C.RustBuffer {
	return LowerIntoRustBuffer[VcChunk](c, value)
}

func (c FfiConverterVcChunk) LowerExternal(value VcChunk) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[VcChunk](c, value))
}

func (c FfiConverterVcChunk) Write(writer io.Writer, value VcChunk) {
	FfiConverterSequenceFloat32INSTANCE.Write(writer, value.Samples)
	FfiConverterBoolINSTANCE.Write(writer, value.IsFinal)
	FfiConverterOptionalFloat32INSTANCE.Write(writer, value.LatencySeconds)
}

type FfiDestroyerVcChunk struct{}

func (_ FfiDestroyerVcChunk) Destroy(value VcChunk) {
	value.Destroy()
}

// A fully-rendered voice-conversion result.
//
// `bytes` carries a complete WAV (RIFF/`fmt `/`data`) container holding
// 16-bit signed little-endian PCM samples at the target voice's native
// sample rate. `sample_rate` echoes that rate for convenience so callers
// don't have to re-parse the WAV header.
type VcResult struct {
	// Encoded audio bytes (WAV container, 16-bit signed PCM).
	Bytes []byte
	// IANA MIME type of `bytes` (always `"audio/wav"` for the native
	// backends shipped today).
	MimeType string
	// Sample rate in Hz, taken from the target voice's
	// [`TargetVoice::sample_rate_hz`].
	SampleRate uint32
	// Duration of the clip in seconds. Zero when the backend did not
	// report one (no extra WAV header parsing happens here).
	DurationSeconds float32
}

func (r *VcResult) Destroy() {
	FfiDestroyerBytes{}.Destroy(r.Bytes)
	FfiDestroyerString{}.Destroy(r.MimeType)
	FfiDestroyerUint32{}.Destroy(r.SampleRate)
	FfiDestroyerFloat32{}.Destroy(r.DurationSeconds)
}

type FfiConverterVcResult struct{}

var FfiConverterVcResultINSTANCE = FfiConverterVcResult{}

func (c FfiConverterVcResult) Lift(rb RustBufferI) VcResult {
	return LiftFromRustBuffer[VcResult](c, rb)
}

func (c FfiConverterVcResult) Read(reader io.Reader) VcResult {
	return VcResult{
		FfiConverterBytesINSTANCE.Read(reader),
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterUint32INSTANCE.Read(reader),
		FfiConverterFloat32INSTANCE.Read(reader),
	}
}

func (c FfiConverterVcResult) Lower(value VcResult) C.RustBuffer {
	return LowerIntoRustBuffer[VcResult](c, value)
}

func (c FfiConverterVcResult) LowerExternal(value VcResult) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[VcResult](c, value))
}

func (c FfiConverterVcResult) Write(writer io.Writer, value VcResult) {
	FfiConverterBytesINSTANCE.Write(writer, value.Bytes)
	FfiConverterStringINSTANCE.Write(writer, value.MimeType)
	FfiConverterUint32INSTANCE.Write(writer, value.SampleRate)
	FfiConverterFloat32INSTANCE.Write(writer, value.DurationSeconds)
}

type FfiDestroyerVcResult struct{}

func (_ FfiDestroyerVcResult) Destroy(value VcResult) {
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

// Local-inference backend identifier returned by
// [`UniffiModelManager::load_from_hf`] and accepted as a forced override on
// [`HfLoadOptionsRecord::backend_hint`].
//
// Mirrors [`blazen_manager::hf_loader::BackendHint`] as a UniFFI Enum.
type BackendHintEnum uint

const (
	// `mistral.rs` — broad architecture coverage, handles both safetensors
	// and GGUF, supports vision/multimodal models.
	BackendHintEnumMistralrs BackendHintEnum = 1
	// `candle` — pure-Rust, supports safetensors and GGUF for the subset of
	// architectures candle ships.
	BackendHintEnumCandle BackendHintEnum = 2
	// `llama.cpp` — GGUF only, best CPU performance and lowest memory.
	BackendHintEnumLlamacpp BackendHintEnum = 3
)

type FfiConverterBackendHintEnum struct{}

var FfiConverterBackendHintEnumINSTANCE = FfiConverterBackendHintEnum{}

func (c FfiConverterBackendHintEnum) Lift(rb RustBufferI) BackendHintEnum {
	return LiftFromRustBuffer[BackendHintEnum](c, rb)
}

func (c FfiConverterBackendHintEnum) Lower(value BackendHintEnum) C.RustBuffer {
	return LowerIntoRustBuffer[BackendHintEnum](c, value)
}

func (c FfiConverterBackendHintEnum) LowerExternal(value BackendHintEnum) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[BackendHintEnum](c, value))
}
func (FfiConverterBackendHintEnum) Read(reader io.Reader) BackendHintEnum {
	id := readInt32(reader)
	return BackendHintEnum(id)
}

func (FfiConverterBackendHintEnum) Write(writer io.Writer, value BackendHintEnum) {
	writeInt32(writer, int32(value))
}

type FfiDestroyerBackendHintEnum struct{}

func (_ FfiDestroyerBackendHintEnum) Destroy(value BackendHintEnum) {
}

// Per-request outcome within a [`BatchResult`].
//
// Slot `i` of [`BatchResult::responses`] corresponds to input request `i`.
// Successful slots carry the [`ModelResponse`]; failed slots carry an
// `error_message` only (the structured `BlazenError` variant doesn't survive
// nesting inside a `uniffi::Enum` cleanly across all four target languages,
// so the message is flattened to a string here — foreign callers wanting
// typed errors should run requests individually).
type BatchItem interface {
	Destroy()
}

// The request completed and the model returned a response.
type BatchItemSuccess struct {
	Response ModelResponse
}

func (e BatchItemSuccess) Destroy() {
	FfiDestroyerModelResponse{}.Destroy(e.Response)
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
			FfiConverterModelResponseINSTANCE.Read(reader),
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
		FfiConverterModelResponseINSTANCE.Write(writer, variant_value.Response)
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
var ErrBlazenErrorCallerError = fmt.Errorf("BlazenErrorCallerError")
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

// Caller-side error raised by a foreign-language tool handler.
//
// Carries structural error data — `name` (foreign-language exception
// class name, e.g. `"SubmitSignal"`), `message`, and `properties_json`
// (JSON-encoded custom attributes). Foreign consumers pattern-match on
// `name` and decode `properties_json` to recover custom payload data.
//
// Full exception class identity is not preserved across the UniFFI
// boundary (the Node/Python/WASM bindings get full `instanceof`
// preservation because they have native object references; UniFFI does
// not). This variant is the structural equivalent.
type BlazenErrorCallerError struct {
	Name           *string
	Message        string
	PropertiesJson string
}

// Caller-side error raised by a foreign-language tool handler.
//
// Carries structural error data — `name` (foreign-language exception
// class name, e.g. `"SubmitSignal"`), `message`, and `properties_json`
// (JSON-encoded custom attributes). Foreign consumers pattern-match on
// `name` and decode `properties_json` to recover custom payload data.
//
// Full exception class identity is not preserved across the UniFFI
// boundary (the Node/Python/WASM bindings get full `instanceof`
// preservation because they have native object references; UniFFI does
// not). This variant is the structural equivalent.
func NewBlazenErrorCallerError(
	name *string,
	message string,
	propertiesJson string,
) *BlazenError {
	return &BlazenError{err: &BlazenErrorCallerError{
		Name:           name,
		Message:        message,
		PropertiesJson: propertiesJson}}
}

func (e BlazenErrorCallerError) destroy() {
	FfiDestroyerOptionalString{}.Destroy(e.Name)
	FfiDestroyerString{}.Destroy(e.Message)
	FfiDestroyerString{}.Destroy(e.PropertiesJson)
}

func (err BlazenErrorCallerError) Error() string {
	return fmt.Sprint("CallerError",
		": ",

		"Name=",
		err.Name,
		", ",
		"Message=",
		err.Message,
		", ",
		"PropertiesJson=",
		err.PropertiesJson,
	)
}

func (self BlazenErrorCallerError) Is(target error) bool {
	return target == ErrBlazenErrorCallerError
}

// Distributed peer-to-peer error and (folded in) distributed
// control-plane error. For peer-mesh failures `kind` is one of
// `"Encode"`, `"Transport"`, `"EnvelopeVersion"`, `"Workflow"`,
// `"Tls"`, `"UnknownStep"`. For control-plane failures `kind` is
// prefixed `"ControlPlane"` (e.g. `"ControlPlaneTransport"`,
// `"ControlPlaneEncode"`, `"ControlPlaneTls"`,
// `"ControlPlaneEnvelopeVersion"`, `"ControlPlaneNoMatchingWorker"`,
// `"ControlPlaneMissingVramHint"`, `"ControlPlaneUnknownRun"`,
// `"ControlPlaneUnknownWorker"`) so foreign consumers can discriminate
// without juggling a second top-level variant.
type BlazenErrorPeer struct {
	Kind    string
	Message string
}

// Distributed peer-to-peer error and (folded in) distributed
// control-plane error. For peer-mesh failures `kind` is one of
// `"Encode"`, `"Transport"`, `"EnvelopeVersion"`, `"Workflow"`,
// `"Tls"`, `"UnknownStep"`. For control-plane failures `kind` is
// prefixed `"ControlPlane"` (e.g. `"ControlPlaneTransport"`,
// `"ControlPlaneEncode"`, `"ControlPlaneTls"`,
// `"ControlPlaneEnvelopeVersion"`, `"ControlPlaneNoMatchingWorker"`,
// `"ControlPlaneMissingVramHint"`, `"ControlPlaneUnknownRun"`,
// `"ControlPlaneUnknownWorker"`) so foreign consumers can discriminate
// without juggling a second top-level variant.
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
		return &BlazenError{&BlazenErrorCallerError{
			Name:           FfiConverterOptionalStringINSTANCE.Read(reader),
			Message:        FfiConverterStringINSTANCE.Read(reader),
			PropertiesJson: FfiConverterStringINSTANCE.Read(reader),
		}}
	case 13:
		return &BlazenError{&BlazenErrorPeer{
			Kind:    FfiConverterStringINSTANCE.Read(reader),
			Message: FfiConverterStringINSTANCE.Read(reader),
		}}
	case 14:
		return &BlazenError{&BlazenErrorPersist{
			Message: FfiConverterStringINSTANCE.Read(reader),
		}}
	case 15:
		return &BlazenError{&BlazenErrorPrompt{
			Kind:    FfiConverterStringINSTANCE.Read(reader),
			Message: FfiConverterStringINSTANCE.Read(reader),
		}}
	case 16:
		return &BlazenError{&BlazenErrorMemory{
			Kind:    FfiConverterStringINSTANCE.Read(reader),
			Message: FfiConverterStringINSTANCE.Read(reader),
		}}
	case 17:
		return &BlazenError{&BlazenErrorCache{
			Kind:    FfiConverterStringINSTANCE.Read(reader),
			Message: FfiConverterStringINSTANCE.Read(reader),
		}}
	case 18:
		return &BlazenError{&BlazenErrorCancelled{}}
	case 19:
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
	case *BlazenErrorCallerError:
		writeInt32(writer, 12)
		FfiConverterOptionalStringINSTANCE.Write(writer, variantValue.Name)
		FfiConverterStringINSTANCE.Write(writer, variantValue.Message)
		FfiConverterStringINSTANCE.Write(writer, variantValue.PropertiesJson)
	case *BlazenErrorPeer:
		writeInt32(writer, 13)
		FfiConverterStringINSTANCE.Write(writer, variantValue.Kind)
		FfiConverterStringINSTANCE.Write(writer, variantValue.Message)
	case *BlazenErrorPersist:
		writeInt32(writer, 14)
		FfiConverterStringINSTANCE.Write(writer, variantValue.Message)
	case *BlazenErrorPrompt:
		writeInt32(writer, 15)
		FfiConverterStringINSTANCE.Write(writer, variantValue.Kind)
		FfiConverterStringINSTANCE.Write(writer, variantValue.Message)
	case *BlazenErrorMemory:
		writeInt32(writer, 16)
		FfiConverterStringINSTANCE.Write(writer, variantValue.Kind)
		FfiConverterStringINSTANCE.Write(writer, variantValue.Message)
	case *BlazenErrorCache:
		writeInt32(writer, 17)
		FfiConverterStringINSTANCE.Write(writer, variantValue.Kind)
		FfiConverterStringINSTANCE.Write(writer, variantValue.Message)
	case *BlazenErrorCancelled:
		writeInt32(writer, 18)
	case *BlazenErrorInternal:
		writeInt32(writer, 19)
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
	case BlazenErrorCallerError:
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

// How a worker declares its admission policy to the control plane.
//
// Carries the union of fields for the three flavours; consumers should
// honour the discriminator in [`ControlPlaneAdmission::mode`].
type ControlPlaneAdmissionMode uint

const (
	// Hard concurrency cap.
	ControlPlaneAdmissionModeFixed ControlPlaneAdmissionMode = 1
	// Worker self-decides via offer/claim/decline.
	ControlPlaneAdmissionModeReactive ControlPlaneAdmissionMode = 2
	// VRAM-sum cap.
	ControlPlaneAdmissionModeVramBudget ControlPlaneAdmissionMode = 3
)

type FfiConverterControlPlaneAdmissionMode struct{}

var FfiConverterControlPlaneAdmissionModeINSTANCE = FfiConverterControlPlaneAdmissionMode{}

func (c FfiConverterControlPlaneAdmissionMode) Lift(rb RustBufferI) ControlPlaneAdmissionMode {
	return LiftFromRustBuffer[ControlPlaneAdmissionMode](c, rb)
}

func (c FfiConverterControlPlaneAdmissionMode) Lower(value ControlPlaneAdmissionMode) C.RustBuffer {
	return LowerIntoRustBuffer[ControlPlaneAdmissionMode](c, value)
}

func (c FfiConverterControlPlaneAdmissionMode) LowerExternal(value ControlPlaneAdmissionMode) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[ControlPlaneAdmissionMode](c, value))
}
func (FfiConverterControlPlaneAdmissionMode) Read(reader io.Reader) ControlPlaneAdmissionMode {
	id := readInt32(reader)
	return ControlPlaneAdmissionMode(id)
}

func (FfiConverterControlPlaneAdmissionMode) Write(writer io.Writer, value ControlPlaneAdmissionMode) {
	writeInt32(writer, int32(value))
}

type FfiDestroyerControlPlaneAdmissionMode struct{}

func (_ FfiDestroyerControlPlaneAdmissionMode) Destroy(value ControlPlaneAdmissionMode) {
}

// Lifecycle state of a workflow run, mirrored across the UniFFI
// boundary.
type ControlPlaneRunStatus uint

const (
	ControlPlaneRunStatusPending   ControlPlaneRunStatus = 1
	ControlPlaneRunStatusRunning   ControlPlaneRunStatus = 2
	ControlPlaneRunStatusCompleted ControlPlaneRunStatus = 3
	ControlPlaneRunStatusFailed    ControlPlaneRunStatus = 4
	ControlPlaneRunStatusCancelled ControlPlaneRunStatus = 5
)

type FfiConverterControlPlaneRunStatus struct{}

var FfiConverterControlPlaneRunStatusINSTANCE = FfiConverterControlPlaneRunStatus{}

func (c FfiConverterControlPlaneRunStatus) Lift(rb RustBufferI) ControlPlaneRunStatus {
	return LiftFromRustBuffer[ControlPlaneRunStatus](c, rb)
}

func (c FfiConverterControlPlaneRunStatus) Lower(value ControlPlaneRunStatus) C.RustBuffer {
	return LowerIntoRustBuffer[ControlPlaneRunStatus](c, value)
}

func (c FfiConverterControlPlaneRunStatus) LowerExternal(value ControlPlaneRunStatus) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[ControlPlaneRunStatus](c, value))
}
func (FfiConverterControlPlaneRunStatus) Read(reader io.Reader) ControlPlaneRunStatus {
	id := readInt32(reader)
	return ControlPlaneRunStatus(id)
}

func (FfiConverterControlPlaneRunStatus) Write(writer io.Writer, value ControlPlaneRunStatus) {
	writeInt32(writer, int32(value))
}

type FfiDestroyerControlPlaneRunStatus struct{}

func (_ FfiDestroyerControlPlaneRunStatus) Destroy(value ControlPlaneRunStatus) {
}

// Backend selector used by [`ModelClient::load_from_hf`]. Mirrors
// [`BackendHintWire`].
type HfBackendHint uint

const (
	// Auto-detect from the repo layout.
	HfBackendHintAuto HfBackendHint = 1
	// Force the `mistral.rs` backend.
	HfBackendHintMistralRs HfBackendHint = 2
	// Force the candle-llm backend.
	HfBackendHintCandle HfBackendHint = 3
	// Force the llama.cpp backend.
	HfBackendHintLlamaCpp HfBackendHint = 4
)

type FfiConverterHfBackendHint struct{}

var FfiConverterHfBackendHintINSTANCE = FfiConverterHfBackendHint{}

func (c FfiConverterHfBackendHint) Lift(rb RustBufferI) HfBackendHint {
	return LiftFromRustBuffer[HfBackendHint](c, rb)
}

func (c FfiConverterHfBackendHint) Lower(value HfBackendHint) C.RustBuffer {
	return LowerIntoRustBuffer[HfBackendHint](c, value)
}

func (c FfiConverterHfBackendHint) LowerExternal(value HfBackendHint) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[HfBackendHint](c, value))
}
func (FfiConverterHfBackendHint) Read(reader io.Reader) HfBackendHint {
	id := readInt32(reader)
	return HfBackendHint(id)
}

func (FfiConverterHfBackendHint) Write(writer io.Writer, value HfBackendHint) {
	writeInt32(writer, int32(value))
}

type FfiDestroyerHfBackendHint struct{}

func (_ FfiDestroyerHfBackendHint) Destroy(value HfBackendHint) {
}

type MixedPrecisionEnum uint

const (
	MixedPrecisionEnumNone MixedPrecisionEnum = 1
	MixedPrecisionEnumBf16 MixedPrecisionEnum = 2
)

type FfiConverterMixedPrecisionEnum struct{}

var FfiConverterMixedPrecisionEnumINSTANCE = FfiConverterMixedPrecisionEnum{}

func (c FfiConverterMixedPrecisionEnum) Lift(rb RustBufferI) MixedPrecisionEnum {
	return LiftFromRustBuffer[MixedPrecisionEnum](c, rb)
}

func (c FfiConverterMixedPrecisionEnum) Lower(value MixedPrecisionEnum) C.RustBuffer {
	return LowerIntoRustBuffer[MixedPrecisionEnum](c, value)
}

func (c FfiConverterMixedPrecisionEnum) LowerExternal(value MixedPrecisionEnum) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[MixedPrecisionEnum](c, value))
}
func (FfiConverterMixedPrecisionEnum) Read(reader io.Reader) MixedPrecisionEnum {
	id := readInt32(reader)
	return MixedPrecisionEnum(id)
}

func (FfiConverterMixedPrecisionEnum) Write(writer io.Writer, value MixedPrecisionEnum) {
	writeInt32(writer, int32(value))
}

type FfiDestroyerMixedPrecisionEnum struct{}

func (_ FfiDestroyerMixedPrecisionEnum) Destroy(value MixedPrecisionEnum) {
}

// Pool a model is registered against. Mirrors
// [`blazen_controlplane::model_protocol::PoolWire`] in a UniFFI-friendly
// shape — Rust's `Gpu(u32)` payload becomes a discriminator + optional
// `device_index` so the enum can cross the FFI boundary cleanly.
type ModelPool interface {
	Destroy()
}

// Host RAM pool.
type ModelPoolCpu struct {
}

func (e ModelPoolCpu) Destroy() {
}

// GPU VRAM pool at `device_index`. Metal collapses to index `0`.
type ModelPoolGpu struct {
	DeviceIndex uint32
}

func (e ModelPoolGpu) Destroy() {
	FfiDestroyerUint32{}.Destroy(e.DeviceIndex)
}

// Off-host pool — memory lives in another process / host.
type ModelPoolRemote struct {
}

func (e ModelPoolRemote) Destroy() {
}

type FfiConverterModelPool struct{}

var FfiConverterModelPoolINSTANCE = FfiConverterModelPool{}

func (c FfiConverterModelPool) Lift(rb RustBufferI) ModelPool {
	return LiftFromRustBuffer[ModelPool](c, rb)
}

func (c FfiConverterModelPool) Lower(value ModelPool) C.RustBuffer {
	return LowerIntoRustBuffer[ModelPool](c, value)
}

func (c FfiConverterModelPool) LowerExternal(value ModelPool) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[ModelPool](c, value))
}
func (FfiConverterModelPool) Read(reader io.Reader) ModelPool {
	id := readInt32(reader)
	switch id {
	case 1:
		return ModelPoolCpu{}
	case 2:
		return ModelPoolGpu{
			FfiConverterUint32INSTANCE.Read(reader),
		}
	case 3:
		return ModelPoolRemote{}
	default:
		panic(fmt.Sprintf("invalid enum value %v in FfiConverterModelPool.Read()", id))
	}
}

func (FfiConverterModelPool) Write(writer io.Writer, value ModelPool) {
	switch variant_value := value.(type) {
	case ModelPoolCpu:
		writeInt32(writer, 1)
	case ModelPoolGpu:
		writeInt32(writer, 2)
		FfiConverterUint32INSTANCE.Write(writer, variant_value.DeviceIndex)
	case ModelPoolRemote:
		writeInt32(writer, 3)
	default:
		_ = variant_value
		panic(fmt.Sprintf("invalid enum value `%v` in FfiConverterModelPool.Write", value))
	}
}

type FfiDestroyerModelPool struct{}

func (_ FfiDestroyerModelPool) Destroy(value ModelPool) {
	value.Destroy()
}

type SchedulerKindEnum uint

const (
	SchedulerKindEnumConstant SchedulerKindEnum = 1
	SchedulerKindEnumLinear   SchedulerKindEnum = 2
	SchedulerKindEnumCosine   SchedulerKindEnum = 3
)

type FfiConverterSchedulerKindEnum struct{}

var FfiConverterSchedulerKindEnumINSTANCE = FfiConverterSchedulerKindEnum{}

func (c FfiConverterSchedulerKindEnum) Lift(rb RustBufferI) SchedulerKindEnum {
	return LiftFromRustBuffer[SchedulerKindEnum](c, rb)
}

func (c FfiConverterSchedulerKindEnum) Lower(value SchedulerKindEnum) C.RustBuffer {
	return LowerIntoRustBuffer[SchedulerKindEnum](c, value)
}

func (c FfiConverterSchedulerKindEnum) LowerExternal(value SchedulerKindEnum) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[SchedulerKindEnum](c, value))
}
func (FfiConverterSchedulerKindEnum) Read(reader io.Reader) SchedulerKindEnum {
	id := readInt32(reader)
	return SchedulerKindEnum(id)
}

func (FfiConverterSchedulerKindEnum) Write(writer io.Writer, value SchedulerKindEnum) {
	writeInt32(writer, int32(value))
}

type FfiDestroyerSchedulerKindEnum struct{}

func (_ FfiDestroyerSchedulerKindEnum) Destroy(value SchedulerKindEnum) {
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

// One observable event emitted during a training run.
type TrainingEventEnum interface {
	Destroy()
}
type TrainingEventEnumStarted struct {
	TotalSteps uint64
}

func (e TrainingEventEnumStarted) Destroy() {
	FfiDestroyerUint64{}.Destroy(e.TotalSteps)
}

type TrainingEventEnumStepCompleted struct {
	Step         uint64
	Loss         float32
	LearningRate float64
	ElapsedMs    uint64
}

func (e TrainingEventEnumStepCompleted) Destroy() {
	FfiDestroyerUint64{}.Destroy(e.Step)
	FfiDestroyerFloat32{}.Destroy(e.Loss)
	FfiDestroyerFloat64{}.Destroy(e.LearningRate)
	FfiDestroyerUint64{}.Destroy(e.ElapsedMs)
}

type TrainingEventEnumEvaluating struct {
	Step uint64
}

func (e TrainingEventEnumEvaluating) Destroy() {
	FfiDestroyerUint64{}.Destroy(e.Step)
}

type TrainingEventEnumEvalCompleted struct {
	Step     uint64
	EvalLoss float32
}

func (e TrainingEventEnumEvalCompleted) Destroy() {
	FfiDestroyerUint64{}.Destroy(e.Step)
	FfiDestroyerFloat32{}.Destroy(e.EvalLoss)
}

type TrainingEventEnumCheckpointSaved struct {
	Step uint64
	Path string
}

func (e TrainingEventEnumCheckpointSaved) Destroy() {
	FfiDestroyerUint64{}.Destroy(e.Step)
	FfiDestroyerString{}.Destroy(e.Path)
}

type TrainingEventEnumFinished struct {
	FinalLoss  float32
	TotalSteps uint64
	AdapterDir string
}

func (e TrainingEventEnumFinished) Destroy() {
	FfiDestroyerFloat32{}.Destroy(e.FinalLoss)
	FfiDestroyerUint64{}.Destroy(e.TotalSteps)
	FfiDestroyerString{}.Destroy(e.AdapterDir)
}

type FfiConverterTrainingEventEnum struct{}

var FfiConverterTrainingEventEnumINSTANCE = FfiConverterTrainingEventEnum{}

func (c FfiConverterTrainingEventEnum) Lift(rb RustBufferI) TrainingEventEnum {
	return LiftFromRustBuffer[TrainingEventEnum](c, rb)
}

func (c FfiConverterTrainingEventEnum) Lower(value TrainingEventEnum) C.RustBuffer {
	return LowerIntoRustBuffer[TrainingEventEnum](c, value)
}

func (c FfiConverterTrainingEventEnum) LowerExternal(value TrainingEventEnum) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[TrainingEventEnum](c, value))
}
func (FfiConverterTrainingEventEnum) Read(reader io.Reader) TrainingEventEnum {
	id := readInt32(reader)
	switch id {
	case 1:
		return TrainingEventEnumStarted{
			FfiConverterUint64INSTANCE.Read(reader),
		}
	case 2:
		return TrainingEventEnumStepCompleted{
			FfiConverterUint64INSTANCE.Read(reader),
			FfiConverterFloat32INSTANCE.Read(reader),
			FfiConverterFloat64INSTANCE.Read(reader),
			FfiConverterUint64INSTANCE.Read(reader),
		}
	case 3:
		return TrainingEventEnumEvaluating{
			FfiConverterUint64INSTANCE.Read(reader),
		}
	case 4:
		return TrainingEventEnumEvalCompleted{
			FfiConverterUint64INSTANCE.Read(reader),
			FfiConverterFloat32INSTANCE.Read(reader),
		}
	case 5:
		return TrainingEventEnumCheckpointSaved{
			FfiConverterUint64INSTANCE.Read(reader),
			FfiConverterStringINSTANCE.Read(reader),
		}
	case 6:
		return TrainingEventEnumFinished{
			FfiConverterFloat32INSTANCE.Read(reader),
			FfiConverterUint64INSTANCE.Read(reader),
			FfiConverterStringINSTANCE.Read(reader),
		}
	default:
		panic(fmt.Sprintf("invalid enum value %v in FfiConverterTrainingEventEnum.Read()", id))
	}
}

func (FfiConverterTrainingEventEnum) Write(writer io.Writer, value TrainingEventEnum) {
	switch variant_value := value.(type) {
	case TrainingEventEnumStarted:
		writeInt32(writer, 1)
		FfiConverterUint64INSTANCE.Write(writer, variant_value.TotalSteps)
	case TrainingEventEnumStepCompleted:
		writeInt32(writer, 2)
		FfiConverterUint64INSTANCE.Write(writer, variant_value.Step)
		FfiConverterFloat32INSTANCE.Write(writer, variant_value.Loss)
		FfiConverterFloat64INSTANCE.Write(writer, variant_value.LearningRate)
		FfiConverterUint64INSTANCE.Write(writer, variant_value.ElapsedMs)
	case TrainingEventEnumEvaluating:
		writeInt32(writer, 3)
		FfiConverterUint64INSTANCE.Write(writer, variant_value.Step)
	case TrainingEventEnumEvalCompleted:
		writeInt32(writer, 4)
		FfiConverterUint64INSTANCE.Write(writer, variant_value.Step)
		FfiConverterFloat32INSTANCE.Write(writer, variant_value.EvalLoss)
	case TrainingEventEnumCheckpointSaved:
		writeInt32(writer, 5)
		FfiConverterUint64INSTANCE.Write(writer, variant_value.Step)
		FfiConverterStringINSTANCE.Write(writer, variant_value.Path)
	case TrainingEventEnumFinished:
		writeInt32(writer, 6)
		FfiConverterFloat32INSTANCE.Write(writer, variant_value.FinalLoss)
		FfiConverterUint64INSTANCE.Write(writer, variant_value.TotalSteps)
		FfiConverterStringINSTANCE.Write(writer, variant_value.AdapterDir)
	default:
		_ = variant_value
		panic(fmt.Sprintf("invalid enum value `%v` in FfiConverterTrainingEventEnum.Write", value))
	}
}

type FfiDestroyerTrainingEventEnum struct{}

func (_ FfiDestroyerTrainingEventEnum) Destroy(value TrainingEventEnum) {
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

type FfiConverterOptionalForeignTrainingProgress struct{}

var FfiConverterOptionalForeignTrainingProgressINSTANCE = FfiConverterOptionalForeignTrainingProgress{}

func (c FfiConverterOptionalForeignTrainingProgress) Lift(rb RustBufferI) *ForeignTrainingProgress {
	return LiftFromRustBuffer[*ForeignTrainingProgress](c, rb)
}

func (_ FfiConverterOptionalForeignTrainingProgress) Read(reader io.Reader) *ForeignTrainingProgress {
	if readInt8(reader) == 0 {
		return nil
	}
	temp := FfiConverterForeignTrainingProgressINSTANCE.Read(reader)
	return &temp
}

func (c FfiConverterOptionalForeignTrainingProgress) Lower(value *ForeignTrainingProgress) C.RustBuffer {
	return LowerIntoRustBuffer[*ForeignTrainingProgress](c, value)
}

func (c FfiConverterOptionalForeignTrainingProgress) LowerExternal(value *ForeignTrainingProgress) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[*ForeignTrainingProgress](c, value))
}

func (_ FfiConverterOptionalForeignTrainingProgress) Write(writer io.Writer, value *ForeignTrainingProgress) {
	if value == nil {
		writeInt8(writer, 0)
	} else {
		writeInt8(writer, 1)
		FfiConverterForeignTrainingProgressINSTANCE.Write(writer, *value)
	}
}

type FfiDestroyerOptionalForeignTrainingProgress struct{}

func (_ FfiDestroyerOptionalForeignTrainingProgress) Destroy(value *ForeignTrainingProgress) {
	if value != nil {
		FfiDestroyerForeignTrainingProgress{}.Destroy(*value)
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

type FfiConverterOptionalBackendHintEnum struct{}

var FfiConverterOptionalBackendHintEnumINSTANCE = FfiConverterOptionalBackendHintEnum{}

func (c FfiConverterOptionalBackendHintEnum) Lift(rb RustBufferI) *BackendHintEnum {
	return LiftFromRustBuffer[*BackendHintEnum](c, rb)
}

func (_ FfiConverterOptionalBackendHintEnum) Read(reader io.Reader) *BackendHintEnum {
	if readInt8(reader) == 0 {
		return nil
	}
	temp := FfiConverterBackendHintEnumINSTANCE.Read(reader)
	return &temp
}

func (c FfiConverterOptionalBackendHintEnum) Lower(value *BackendHintEnum) C.RustBuffer {
	return LowerIntoRustBuffer[*BackendHintEnum](c, value)
}

func (c FfiConverterOptionalBackendHintEnum) LowerExternal(value *BackendHintEnum) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[*BackendHintEnum](c, value))
}

func (_ FfiConverterOptionalBackendHintEnum) Write(writer io.Writer, value *BackendHintEnum) {
	if value == nil {
		writeInt8(writer, 0)
	} else {
		writeInt8(writer, 1)
		FfiConverterBackendHintEnumINSTANCE.Write(writer, *value)
	}
}

type FfiDestroyerOptionalBackendHintEnum struct{}

func (_ FfiDestroyerOptionalBackendHintEnum) Destroy(value *BackendHintEnum) {
	if value != nil {
		FfiDestroyerBackendHintEnum{}.Destroy(*value)
	}
}

type FfiConverterOptionalHfBackendHint struct{}

var FfiConverterOptionalHfBackendHintINSTANCE = FfiConverterOptionalHfBackendHint{}

func (c FfiConverterOptionalHfBackendHint) Lift(rb RustBufferI) *HfBackendHint {
	return LiftFromRustBuffer[*HfBackendHint](c, rb)
}

func (_ FfiConverterOptionalHfBackendHint) Read(reader io.Reader) *HfBackendHint {
	if readInt8(reader) == 0 {
		return nil
	}
	temp := FfiConverterHfBackendHintINSTANCE.Read(reader)
	return &temp
}

func (c FfiConverterOptionalHfBackendHint) Lower(value *HfBackendHint) C.RustBuffer {
	return LowerIntoRustBuffer[*HfBackendHint](c, value)
}

func (c FfiConverterOptionalHfBackendHint) LowerExternal(value *HfBackendHint) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[*HfBackendHint](c, value))
}

func (_ FfiConverterOptionalHfBackendHint) Write(writer io.Writer, value *HfBackendHint) {
	if value == nil {
		writeInt8(writer, 0)
	} else {
		writeInt8(writer, 1)
		FfiConverterHfBackendHintINSTANCE.Write(writer, *value)
	}
}

type FfiDestroyerOptionalHfBackendHint struct{}

func (_ FfiDestroyerOptionalHfBackendHint) Destroy(value *HfBackendHint) {
	if value != nil {
		FfiDestroyerHfBackendHint{}.Destroy(*value)
	}
}

type FfiConverterSequenceFloat32 struct{}

var FfiConverterSequenceFloat32INSTANCE = FfiConverterSequenceFloat32{}

func (c FfiConverterSequenceFloat32) Lift(rb RustBufferI) []float32 {
	return LiftFromRustBuffer[[]float32](c, rb)
}

func (c FfiConverterSequenceFloat32) Read(reader io.Reader) []float32 {
	length := readInt32(reader)
	if length == 0 {
		return nil
	}
	result := make([]float32, 0, length)
	for i := int32(0); i < length; i++ {
		result = append(result, FfiConverterFloat32INSTANCE.Read(reader))
	}
	return result
}

func (c FfiConverterSequenceFloat32) Lower(value []float32) C.RustBuffer {
	return LowerIntoRustBuffer[[]float32](c, value)
}

func (c FfiConverterSequenceFloat32) LowerExternal(value []float32) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[[]float32](c, value))
}

func (c FfiConverterSequenceFloat32) Write(writer io.Writer, value []float32) {
	if len(value) > math.MaxInt32 {
		panic("[]float32 is too large to fit into Int32")
	}

	writeInt32(writer, int32(len(value)))
	for _, item := range value {
		FfiConverterFloat32INSTANCE.Write(writer, item)
	}
}

type FfiDestroyerSequenceFloat32 struct{}

func (FfiDestroyerSequenceFloat32) Destroy(sequence []float32) {
	for _, value := range sequence {
		FfiDestroyerFloat32{}.Destroy(value)
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

type FfiConverterSequenceAdapterStatusRecord struct{}

var FfiConverterSequenceAdapterStatusRecordINSTANCE = FfiConverterSequenceAdapterStatusRecord{}

func (c FfiConverterSequenceAdapterStatusRecord) Lift(rb RustBufferI) []AdapterStatusRecord {
	return LiftFromRustBuffer[[]AdapterStatusRecord](c, rb)
}

func (c FfiConverterSequenceAdapterStatusRecord) Read(reader io.Reader) []AdapterStatusRecord {
	length := readInt32(reader)
	if length == 0 {
		return nil
	}
	result := make([]AdapterStatusRecord, 0, length)
	for i := int32(0); i < length; i++ {
		result = append(result, FfiConverterAdapterStatusRecordINSTANCE.Read(reader))
	}
	return result
}

func (c FfiConverterSequenceAdapterStatusRecord) Lower(value []AdapterStatusRecord) C.RustBuffer {
	return LowerIntoRustBuffer[[]AdapterStatusRecord](c, value)
}

func (c FfiConverterSequenceAdapterStatusRecord) LowerExternal(value []AdapterStatusRecord) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[[]AdapterStatusRecord](c, value))
}

func (c FfiConverterSequenceAdapterStatusRecord) Write(writer io.Writer, value []AdapterStatusRecord) {
	if len(value) > math.MaxInt32 {
		panic("[]AdapterStatusRecord is too large to fit into Int32")
	}

	writeInt32(writer, int32(len(value)))
	for _, item := range value {
		FfiConverterAdapterStatusRecordINSTANCE.Write(writer, item)
	}
}

type FfiDestroyerSequenceAdapterStatusRecord struct{}

func (FfiDestroyerSequenceAdapterStatusRecord) Destroy(sequence []AdapterStatusRecord) {
	for _, value := range sequence {
		FfiDestroyerAdapterStatusRecord{}.Destroy(value)
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

type FfiConverterSequenceControlPlaneWorkerCapability struct{}

var FfiConverterSequenceControlPlaneWorkerCapabilityINSTANCE = FfiConverterSequenceControlPlaneWorkerCapability{}

func (c FfiConverterSequenceControlPlaneWorkerCapability) Lift(rb RustBufferI) []ControlPlaneWorkerCapability {
	return LiftFromRustBuffer[[]ControlPlaneWorkerCapability](c, rb)
}

func (c FfiConverterSequenceControlPlaneWorkerCapability) Read(reader io.Reader) []ControlPlaneWorkerCapability {
	length := readInt32(reader)
	if length == 0 {
		return nil
	}
	result := make([]ControlPlaneWorkerCapability, 0, length)
	for i := int32(0); i < length; i++ {
		result = append(result, FfiConverterControlPlaneWorkerCapabilityINSTANCE.Read(reader))
	}
	return result
}

func (c FfiConverterSequenceControlPlaneWorkerCapability) Lower(value []ControlPlaneWorkerCapability) C.RustBuffer {
	return LowerIntoRustBuffer[[]ControlPlaneWorkerCapability](c, value)
}

func (c FfiConverterSequenceControlPlaneWorkerCapability) LowerExternal(value []ControlPlaneWorkerCapability) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[[]ControlPlaneWorkerCapability](c, value))
}

func (c FfiConverterSequenceControlPlaneWorkerCapability) Write(writer io.Writer, value []ControlPlaneWorkerCapability) {
	if len(value) > math.MaxInt32 {
		panic("[]ControlPlaneWorkerCapability is too large to fit into Int32")
	}

	writeInt32(writer, int32(len(value)))
	for _, item := range value {
		FfiConverterControlPlaneWorkerCapabilityINSTANCE.Write(writer, item)
	}
}

type FfiDestroyerSequenceControlPlaneWorkerCapability struct{}

func (FfiDestroyerSequenceControlPlaneWorkerCapability) Destroy(sequence []ControlPlaneWorkerCapability) {
	for _, value := range sequence {
		FfiDestroyerControlPlaneWorkerCapability{}.Destroy(value)
	}
}

type FfiConverterSequenceControlPlaneWorkerInfo struct{}

var FfiConverterSequenceControlPlaneWorkerInfoINSTANCE = FfiConverterSequenceControlPlaneWorkerInfo{}

func (c FfiConverterSequenceControlPlaneWorkerInfo) Lift(rb RustBufferI) []ControlPlaneWorkerInfo {
	return LiftFromRustBuffer[[]ControlPlaneWorkerInfo](c, rb)
}

func (c FfiConverterSequenceControlPlaneWorkerInfo) Read(reader io.Reader) []ControlPlaneWorkerInfo {
	length := readInt32(reader)
	if length == 0 {
		return nil
	}
	result := make([]ControlPlaneWorkerInfo, 0, length)
	for i := int32(0); i < length; i++ {
		result = append(result, FfiConverterControlPlaneWorkerInfoINSTANCE.Read(reader))
	}
	return result
}

func (c FfiConverterSequenceControlPlaneWorkerInfo) Lower(value []ControlPlaneWorkerInfo) C.RustBuffer {
	return LowerIntoRustBuffer[[]ControlPlaneWorkerInfo](c, value)
}

func (c FfiConverterSequenceControlPlaneWorkerInfo) LowerExternal(value []ControlPlaneWorkerInfo) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[[]ControlPlaneWorkerInfo](c, value))
}

func (c FfiConverterSequenceControlPlaneWorkerInfo) Write(writer io.Writer, value []ControlPlaneWorkerInfo) {
	if len(value) > math.MaxInt32 {
		panic("[]ControlPlaneWorkerInfo is too large to fit into Int32")
	}

	writeInt32(writer, int32(len(value)))
	for _, item := range value {
		FfiConverterControlPlaneWorkerInfoINSTANCE.Write(writer, item)
	}
}

type FfiDestroyerSequenceControlPlaneWorkerInfo struct{}

func (FfiDestroyerSequenceControlPlaneWorkerInfo) Destroy(sequence []ControlPlaneWorkerInfo) {
	for _, value := range sequence {
		FfiDestroyerControlPlaneWorkerInfo{}.Destroy(value)
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

type FfiConverterSequenceModelClientStatusRecord struct{}

var FfiConverterSequenceModelClientStatusRecordINSTANCE = FfiConverterSequenceModelClientStatusRecord{}

func (c FfiConverterSequenceModelClientStatusRecord) Lift(rb RustBufferI) []ModelClientStatusRecord {
	return LiftFromRustBuffer[[]ModelClientStatusRecord](c, rb)
}

func (c FfiConverterSequenceModelClientStatusRecord) Read(reader io.Reader) []ModelClientStatusRecord {
	length := readInt32(reader)
	if length == 0 {
		return nil
	}
	result := make([]ModelClientStatusRecord, 0, length)
	for i := int32(0); i < length; i++ {
		result = append(result, FfiConverterModelClientStatusRecordINSTANCE.Read(reader))
	}
	return result
}

func (c FfiConverterSequenceModelClientStatusRecord) Lower(value []ModelClientStatusRecord) C.RustBuffer {
	return LowerIntoRustBuffer[[]ModelClientStatusRecord](c, value)
}

func (c FfiConverterSequenceModelClientStatusRecord) LowerExternal(value []ModelClientStatusRecord) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[[]ModelClientStatusRecord](c, value))
}

func (c FfiConverterSequenceModelClientStatusRecord) Write(writer io.Writer, value []ModelClientStatusRecord) {
	if len(value) > math.MaxInt32 {
		panic("[]ModelClientStatusRecord is too large to fit into Int32")
	}

	writeInt32(writer, int32(len(value)))
	for _, item := range value {
		FfiConverterModelClientStatusRecordINSTANCE.Write(writer, item)
	}
}

type FfiDestroyerSequenceModelClientStatusRecord struct{}

func (FfiDestroyerSequenceModelClientStatusRecord) Destroy(sequence []ModelClientStatusRecord) {
	for _, value := range sequence {
		FfiDestroyerModelClientStatusRecord{}.Destroy(value)
	}
}

type FfiConverterSequenceModelRequest struct{}

var FfiConverterSequenceModelRequestINSTANCE = FfiConverterSequenceModelRequest{}

func (c FfiConverterSequenceModelRequest) Lift(rb RustBufferI) []ModelRequest {
	return LiftFromRustBuffer[[]ModelRequest](c, rb)
}

func (c FfiConverterSequenceModelRequest) Read(reader io.Reader) []ModelRequest {
	length := readInt32(reader)
	if length == 0 {
		return nil
	}
	result := make([]ModelRequest, 0, length)
	for i := int32(0); i < length; i++ {
		result = append(result, FfiConverterModelRequestINSTANCE.Read(reader))
	}
	return result
}

func (c FfiConverterSequenceModelRequest) Lower(value []ModelRequest) C.RustBuffer {
	return LowerIntoRustBuffer[[]ModelRequest](c, value)
}

func (c FfiConverterSequenceModelRequest) LowerExternal(value []ModelRequest) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[[]ModelRequest](c, value))
}

func (c FfiConverterSequenceModelRequest) Write(writer io.Writer, value []ModelRequest) {
	if len(value) > math.MaxInt32 {
		panic("[]ModelRequest is too large to fit into Int32")
	}

	writeInt32(writer, int32(len(value)))
	for _, item := range value {
		FfiConverterModelRequestINSTANCE.Write(writer, item)
	}
}

type FfiDestroyerSequenceModelRequest struct{}

func (FfiDestroyerSequenceModelRequest) Destroy(sequence []ModelRequest) {
	for _, value := range sequence {
		FfiDestroyerModelRequest{}.Destroy(value)
	}
}

type FfiConverterSequenceModelStatusRecord struct{}

var FfiConverterSequenceModelStatusRecordINSTANCE = FfiConverterSequenceModelStatusRecord{}

func (c FfiConverterSequenceModelStatusRecord) Lift(rb RustBufferI) []ModelStatusRecord {
	return LiftFromRustBuffer[[]ModelStatusRecord](c, rb)
}

func (c FfiConverterSequenceModelStatusRecord) Read(reader io.Reader) []ModelStatusRecord {
	length := readInt32(reader)
	if length == 0 {
		return nil
	}
	result := make([]ModelStatusRecord, 0, length)
	for i := int32(0); i < length; i++ {
		result = append(result, FfiConverterModelStatusRecordINSTANCE.Read(reader))
	}
	return result
}

func (c FfiConverterSequenceModelStatusRecord) Lower(value []ModelStatusRecord) C.RustBuffer {
	return LowerIntoRustBuffer[[]ModelStatusRecord](c, value)
}

func (c FfiConverterSequenceModelStatusRecord) LowerExternal(value []ModelStatusRecord) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[[]ModelStatusRecord](c, value))
}

func (c FfiConverterSequenceModelStatusRecord) Write(writer io.Writer, value []ModelStatusRecord) {
	if len(value) > math.MaxInt32 {
		panic("[]ModelStatusRecord is too large to fit into Int32")
	}

	writeInt32(writer, int32(len(value)))
	for _, item := range value {
		FfiConverterModelStatusRecordINSTANCE.Write(writer, item)
	}
}

type FfiDestroyerSequenceModelStatusRecord struct{}

func (FfiDestroyerSequenceModelStatusRecord) Destroy(sequence []ModelStatusRecord) {
	for _, value := range sequence {
		FfiDestroyerModelStatusRecord{}.Destroy(value)
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

type FfiConverterSequencePoolStatusRecord struct{}

var FfiConverterSequencePoolStatusRecordINSTANCE = FfiConverterSequencePoolStatusRecord{}

func (c FfiConverterSequencePoolStatusRecord) Lift(rb RustBufferI) []PoolStatusRecord {
	return LiftFromRustBuffer[[]PoolStatusRecord](c, rb)
}

func (c FfiConverterSequencePoolStatusRecord) Read(reader io.Reader) []PoolStatusRecord {
	length := readInt32(reader)
	if length == 0 {
		return nil
	}
	result := make([]PoolStatusRecord, 0, length)
	for i := int32(0); i < length; i++ {
		result = append(result, FfiConverterPoolStatusRecordINSTANCE.Read(reader))
	}
	return result
}

func (c FfiConverterSequencePoolStatusRecord) Lower(value []PoolStatusRecord) C.RustBuffer {
	return LowerIntoRustBuffer[[]PoolStatusRecord](c, value)
}

func (c FfiConverterSequencePoolStatusRecord) LowerExternal(value []PoolStatusRecord) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[[]PoolStatusRecord](c, value))
}

func (c FfiConverterSequencePoolStatusRecord) Write(writer io.Writer, value []PoolStatusRecord) {
	if len(value) > math.MaxInt32 {
		panic("[]PoolStatusRecord is too large to fit into Int32")
	}

	writeInt32(writer, int32(len(value)))
	for _, item := range value {
		FfiConverterPoolStatusRecordINSTANCE.Write(writer, item)
	}
}

type FfiDestroyerSequencePoolStatusRecord struct{}

func (FfiDestroyerSequencePoolStatusRecord) Destroy(sequence []PoolStatusRecord) {
	for _, value := range sequence {
		FfiDestroyerPoolStatusRecord{}.Destroy(value)
	}
}

type FfiConverterSequenceTargetVoice struct{}

var FfiConverterSequenceTargetVoiceINSTANCE = FfiConverterSequenceTargetVoice{}

func (c FfiConverterSequenceTargetVoice) Lift(rb RustBufferI) []TargetVoice {
	return LiftFromRustBuffer[[]TargetVoice](c, rb)
}

func (c FfiConverterSequenceTargetVoice) Read(reader io.Reader) []TargetVoice {
	length := readInt32(reader)
	if length == 0 {
		return nil
	}
	result := make([]TargetVoice, 0, length)
	for i := int32(0); i < length; i++ {
		result = append(result, FfiConverterTargetVoiceINSTANCE.Read(reader))
	}
	return result
}

func (c FfiConverterSequenceTargetVoice) Lower(value []TargetVoice) C.RustBuffer {
	return LowerIntoRustBuffer[[]TargetVoice](c, value)
}

func (c FfiConverterSequenceTargetVoice) LowerExternal(value []TargetVoice) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[[]TargetVoice](c, value))
}

func (c FfiConverterSequenceTargetVoice) Write(writer io.Writer, value []TargetVoice) {
	if len(value) > math.MaxInt32 {
		panic("[]TargetVoice is too large to fit into Int32")
	}

	writeInt32(writer, int32(len(value)))
	for _, item := range value {
		FfiConverterTargetVoiceINSTANCE.Write(writer, item)
	}
}

type FfiDestroyerSequenceTargetVoice struct{}

func (FfiDestroyerSequenceTargetVoice) Destroy(sequence []TargetVoice) {
	for _, value := range sequence {
		FfiDestroyerTargetVoice{}.Destroy(value)
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

type FfiConverterMapStringFloat64 struct{}

var FfiConverterMapStringFloat64INSTANCE = FfiConverterMapStringFloat64{}

func (c FfiConverterMapStringFloat64) Lift(rb RustBufferI) map[string]float64 {
	return LiftFromRustBuffer[map[string]float64](c, rb)
}

func (_ FfiConverterMapStringFloat64) Read(reader io.Reader) map[string]float64 {
	result := make(map[string]float64)
	length := readInt32(reader)
	for i := int32(0); i < length; i++ {
		key := FfiConverterStringINSTANCE.Read(reader)
		value := FfiConverterFloat64INSTANCE.Read(reader)
		result[key] = value
	}
	return result
}

func (c FfiConverterMapStringFloat64) Lower(value map[string]float64) C.RustBuffer {
	return LowerIntoRustBuffer[map[string]float64](c, value)
}

func (c FfiConverterMapStringFloat64) LowerExternal(value map[string]float64) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[map[string]float64](c, value))
}

func (_ FfiConverterMapStringFloat64) Write(writer io.Writer, mapValue map[string]float64) {
	if len(mapValue) > math.MaxInt32 {
		panic("map[string]float64 is too large to fit into Int32")
	}

	writeInt32(writer, int32(len(mapValue)))
	for key, value := range mapValue {
		FfiConverterStringINSTANCE.Write(writer, key)
		FfiConverterFloat64INSTANCE.Write(writer, value)
	}
}

type FfiDestroyerMapStringFloat64 struct{}

func (_ FfiDestroyerMapStringFloat64) Destroy(mapValue map[string]float64) {
	for key, value := range mapValue {
		FfiDestroyerString{}.Destroy(key)
		FfiDestroyerFloat64{}.Destroy(value)
	}
}

type FfiConverterMapStringString struct{}

var FfiConverterMapStringStringINSTANCE = FfiConverterMapStringString{}

func (c FfiConverterMapStringString) Lift(rb RustBufferI) map[string]string {
	return LiftFromRustBuffer[map[string]string](c, rb)
}

func (_ FfiConverterMapStringString) Read(reader io.Reader) map[string]string {
	result := make(map[string]string)
	length := readInt32(reader)
	for i := int32(0); i < length; i++ {
		key := FfiConverterStringINSTANCE.Read(reader)
		value := FfiConverterStringINSTANCE.Read(reader)
		result[key] = value
	}
	return result
}

func (c FfiConverterMapStringString) Lower(value map[string]string) C.RustBuffer {
	return LowerIntoRustBuffer[map[string]string](c, value)
}

func (c FfiConverterMapStringString) LowerExternal(value map[string]string) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[map[string]string](c, value))
}

func (_ FfiConverterMapStringString) Write(writer io.Writer, mapValue map[string]string) {
	if len(mapValue) > math.MaxInt32 {
		panic("map[string]string is too large to fit into Int32")
	}

	writeInt32(writer, int32(len(mapValue)))
	for key, value := range mapValue {
		FfiConverterStringINSTANCE.Write(writer, key)
		FfiConverterStringINSTANCE.Write(writer, value)
	}
}

type FfiDestroyerMapStringString struct{}

func (_ FfiDestroyerMapStringString) Destroy(mapValue map[string]string) {
	for key, value := range mapValue {
		FfiDestroyerString{}.Destroy(key)
		FfiDestroyerString{}.Destroy(value)
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
func CompleteBatch(model *Model, requests []ModelRequest, maxConcurrency uint32) (BatchResult, error) {
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
		C.uniffi_blazen_uniffi_fn_func_complete_batch(FfiConverterModelINSTANCE.Lower(model), FfiConverterSequenceModelRequestINSTANCE.Lower(requests), FfiConverterUint32INSTANCE.Lower(maxConcurrency)),
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
func CompleteBatchBlocking(model *Model, requests []ModelRequest, maxConcurrency uint32) (BatchResult, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) RustBufferI {
		return GoRustBuffer{
			inner: C.uniffi_blazen_uniffi_fn_func_complete_batch_blocking(FfiConverterModelINSTANCE.Lower(model), FfiConverterSequenceModelRequestINSTANCE.Lower(requests), FfiConverterUint32INSTANCE.Lower(maxConcurrency), _uniffiStatus),
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

// Build a local TTS model backed by `any-tts`.
//
// `model` is one of `"kokoro82m"`, `"vibevoice"`, or `"qwen3_tts"` (or
// any of the snake_case aliases); pass null to default to Kokoro-82M.
// `voice` selects a speaker preset (e.g. `"af_bella"`); pass null to
// use the model default. `sample_rate` overrides the model's native
// sample rate.
func NewLocalTtsModel(model *string, voice *string, language *string, sampleRate *uint32) (*TtsModel, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_local_tts_model(FfiConverterOptionalStringINSTANCE.Lower(model), FfiConverterOptionalStringINSTANCE.Lower(voice), FfiConverterOptionalStringINSTANCE.Lower(language), FfiConverterOptionalUint32INSTANCE.Lower(sampleRate), _uniffiStatus)
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
// `model_id` is a Piper voice id like `"en_US-amy-medium"` — this is
// resolved to the `rhasspy/piper-voices` repo path
// `en/en_US/amy/medium/en_US-amy-medium.onnx[.json]` and the two files
// are downloaded (or read from cache) before the backend is built.
//
// `speaker_id` is forwarded to the Piper ONNX session for
// multi-speaker voices (e.g. `en_US-libritts_r-medium` exposes 904
// speakers). `None` defaults to speaker 0 / the voice's single
// speaker.
//
// `sample_rate` is reserved; the Piper voice file is authoritative
// for the output sample rate. If provided, it is logged at trace
// level and otherwise ignored.
func NewPiperTtsModel(modelId string, speakerId *uint32, sampleRate *uint32) (*TtsModel, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_piper_tts_model(FfiConverterStringINSTANCE.Lower(modelId), FfiConverterOptionalUint32INSTANCE.Lower(speakerId), FfiConverterOptionalUint32INSTANCE.Lower(sampleRate), _uniffiStatus)
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

// Build a native AudioGen-backed [`MusicModel`].
//
// `repo_id` overrides the default Hugging Face repo (defaults to
// `facebook/audiogen-medium`). `revision` pins a specific commit / tag.
// `device` / `cache_dir` / `max_duration_seconds` follow the MusicGen
// factory's conventions.
func NewAudiogenModel(repoId *string, revision *string, device *string, cacheDir *string, maxDurationSeconds *float32) (*MusicModel, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_audiogen_model(FfiConverterOptionalStringINSTANCE.Lower(repoId), FfiConverterOptionalStringINSTANCE.Lower(revision), FfiConverterOptionalStringINSTANCE.Lower(device), FfiConverterOptionalStringINSTANCE.Lower(cacheDir), FfiConverterOptionalFloat32INSTANCE.Lower(maxDurationSeconds), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *MusicModel
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterMusicModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build a fal.ai-backed [`MusicModel`].
//
// `api_key` may be empty when the provider resolves it from `FAL_KEY`.
// `model` overrides the default fal music / SFX endpoint (the same
// override is applied to both `generate_music` and `generate_sfx` calls
// — fal's per-endpoint dispatch handles the routing).
func NewFalMusicModel(apiKey string, model *string) (*MusicModel, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_fal_music_model(FfiConverterStringINSTANCE.Lower(apiKey), FfiConverterOptionalStringINSTANCE.Lower(model), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *MusicModel
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterMusicModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build a native MusicGen-backed [`MusicModel`].
//
// `variant` selects the MusicGen checkpoint by name (case-insensitive:
// `"small"`, `"medium"`, `"large"`); unrecognised values default to
// `Small`. `device` accepts the same format strings as
// `blazen_llm::Device::parse` (`"cpu"`, `"cuda"`, `"cuda:N"`, `"metal"`);
// `None` defers to the backend's auto-detection (CUDA → Metal → CPU).
// `cache_dir` overrides the Hugging Face Hub cache directory.
// `max_duration_seconds` overrides the default 30 s per-call safety cap
// (hard ceiling stays at `MUSICGEN_MAX_DURATION_HARD_LIMIT`).
func NewMusicgenModel(variant *string, device *string, cacheDir *string, maxDurationSeconds *float32) (*MusicModel, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_musicgen_model(FfiConverterOptionalStringINSTANCE.Lower(variant), FfiConverterOptionalStringINSTANCE.Lower(device), FfiConverterOptionalStringINSTANCE.Lower(cacheDir), FfiConverterOptionalFloat32INSTANCE.Lower(maxDurationSeconds), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *MusicModel
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterMusicModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build a native Stable Audio Open-backed [`MusicModel`].
//
// `variant` selects the Stable Audio Open checkpoint by name
// (case-insensitive: `"small"`, `"open-1.0"` / `"open1.0"`); unrecognised
// values default to `Small`. `tokenizer_path` must point at the T5
// SentencePiece `tokenizer.json` shipped with the Stable Audio Open repo
// — required because Stable Audio's tokenizer is not auto-downloaded by
// the backend today. `device` follows the same device-string format as
// the MusicGen factory. `max_duration_seconds` is accepted for API
// symmetry but Stable Audio enforces its own variant-dependent ceiling
// internally.
func NewStableAudioModel(variant *string, tokenizerPath string, device *string, maxDurationSeconds *float32) (*MusicModel, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_stable_audio_model(FfiConverterOptionalStringINSTANCE.Lower(variant), FfiConverterStringINSTANCE.Lower(tokenizerPath), FfiConverterOptionalStringINSTANCE.Lower(device), FfiConverterOptionalFloat32INSTANCE.Lower(maxDurationSeconds), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *MusicModel
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterMusicModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Drive a streaming music-generation call, dispatching each chunk to the
// sink.
//
// On success, calls `sink.on_done()` exactly once and returns `Ok(())`.
// On a backend-side or sink-side failure, calls `sink.on_error(...)` and
// returns `Ok(())` — error delivery is the sink's responsibility, matching
// the convention `complete_streaming` established for chat completions.
//
// The only failure mode that propagates back to the caller is a panic in
// the sink itself or the runtime; init errors (e.g. fal.ai not supporting
// streaming, MusicGen weight-download failure) are delivered through
// `on_error`.
func StreamGenerateMusicToSink(model *MusicModel, prompt string, durationSeconds float32, sink MusicStreamSink) error {
	_, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) struct{} {
			C.ffi_blazen_uniffi_rust_future_complete_void(handle, status)
			return struct{}{}
		},
		// liftFn
		func(_ struct{}) struct{} { return struct{}{} },
		C.uniffi_blazen_uniffi_fn_func_stream_generate_music_to_sink(FfiConverterMusicModelINSTANCE.Lower(model), FfiConverterStringINSTANCE.Lower(prompt), FfiConverterFloat32INSTANCE.Lower(durationSeconds), FfiConverterMusicStreamSinkINSTANCE.Lower(sink)),
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

// Synchronous variant of [`stream_generate_music_to_sink`] — blocks the
// current thread on the shared Tokio runtime.
func StreamGenerateMusicToSinkBlocking(model *MusicModel, prompt string, durationSeconds float32, sink MusicStreamSink) error {
	_, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) bool {
		C.uniffi_blazen_uniffi_fn_func_stream_generate_music_to_sink_blocking(FfiConverterMusicModelINSTANCE.Lower(model), FfiConverterStringINSTANCE.Lower(prompt), FfiConverterFloat32INSTANCE.Lower(durationSeconds), FfiConverterMusicStreamSinkINSTANCE.Lower(sink), _uniffiStatus)
		return false
	})
	return _uniffiErr.AsError()
}

// Drive a streaming SFX-generation call, dispatching each chunk to the
// sink. Same semantics as [`stream_generate_music_to_sink`].
func StreamGenerateSfxToSink(model *MusicModel, prompt string, durationSeconds float32, sink MusicStreamSink) error {
	_, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) struct{} {
			C.ffi_blazen_uniffi_rust_future_complete_void(handle, status)
			return struct{}{}
		},
		// liftFn
		func(_ struct{}) struct{} { return struct{}{} },
		C.uniffi_blazen_uniffi_fn_func_stream_generate_sfx_to_sink(FfiConverterMusicModelINSTANCE.Lower(model), FfiConverterStringINSTANCE.Lower(prompt), FfiConverterFloat32INSTANCE.Lower(durationSeconds), FfiConverterMusicStreamSinkINSTANCE.Lower(sink)),
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

// Synchronous variant of [`stream_generate_sfx_to_sink`] — blocks the
// current thread on the shared Tokio runtime.
func StreamGenerateSfxToSinkBlocking(model *MusicModel, prompt string, durationSeconds float32, sink MusicStreamSink) error {
	_, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) bool {
		C.uniffi_blazen_uniffi_fn_func_stream_generate_sfx_to_sink_blocking(FfiConverterMusicModelINSTANCE.Lower(model), FfiConverterStringINSTANCE.Lower(prompt), FfiConverterFloat32INSTANCE.Lower(durationSeconds), FfiConverterMusicStreamSinkINSTANCE.Lower(sink), _uniffiStatus)
		return false
	})
	return _uniffiErr.AsError()
}

// Build a native RVC-backed [`VcModel`].
//
// `voice_dir` overrides the per-process `BLAZEN_RVC_VOICE_DIR`
// environment variable that the RVC pipeline reads to locate voice
// profiles on disk (each voice is expected to live at
// `<voice_dir>/<voice_id>/` with `model.pth`, `index.index`, and
// `metadata.json`). When `None`, the existing process-environment value
// is used unchanged. Setting this from inside the factory mutates global
// process state via `std::env::set_var` — callers running multiple RVC
// instances in the same process should pick a single voice directory
// rather than racing factory calls.
//
// `device` accepts the same format strings as `blazen_llm::Device::parse`
// (`"cpu"`, `"cuda"`, `"cuda:N"`, `"metal"`); `None` defers to CPU.
func NewRvcModel(voiceDir *string, device *string) (*VcModel, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_rvc_model(FfiConverterOptionalStringINSTANCE.Lower(voiceDir), FfiConverterOptionalStringINSTANCE.Lower(device), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *VcModel
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterVcModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Drive a streaming voice-conversion call, dispatching each chunk to the
// sink.
//
// `input_pcm` is the full source utterance as 32-bit float PCM at the
// backend's expected source sample rate (typically 16 kHz mono for RVC).
//
// On success, calls `sink.on_done()` exactly once and returns `Ok(())`.
// On a backend-side or sink-side failure, calls `sink.on_error(...)` and
// returns `Ok(())` — error delivery is the sink's responsibility, matching
// the convention `complete_streaming` and `stream_generate_music_to_sink`
// established.
//
// The only failure mode that propagates back to the caller is a panic in
// the sink itself or the runtime; init errors (e.g. voice-not-found,
// backend-not-built-with-feature) are delivered through `on_error`.
func StreamConvertPcmToSink(model *VcModel, inputPcm []float32, targetVoiceId string, sink VcStreamSink) error {
	_, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) struct{} {
			C.ffi_blazen_uniffi_rust_future_complete_void(handle, status)
			return struct{}{}
		},
		// liftFn
		func(_ struct{}) struct{} { return struct{}{} },
		C.uniffi_blazen_uniffi_fn_func_stream_convert_pcm_to_sink(FfiConverterVcModelINSTANCE.Lower(model), FfiConverterSequenceFloat32INSTANCE.Lower(inputPcm), FfiConverterStringINSTANCE.Lower(targetVoiceId), FfiConverterVcStreamSinkINSTANCE.Lower(sink)),
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

// Synchronous variant of [`stream_convert_pcm_to_sink`] — blocks the
// current thread on the shared Tokio runtime.
func StreamConvertPcmToSinkBlocking(model *VcModel, inputPcm []float32, targetVoiceId string, sink VcStreamSink) error {
	_, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) bool {
		C.uniffi_blazen_uniffi_fn_func_stream_convert_pcm_to_sink_blocking(FfiConverterVcModelINSTANCE.Lower(model), FfiConverterSequenceFloat32INSTANCE.Lower(inputPcm), FfiConverterStringINSTANCE.Lower(targetVoiceId), FfiConverterVcStreamSinkINSTANCE.Lower(sink), _uniffiStatus)
		return false
	})
	return _uniffiErr.AsError()
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

// Refresh the pricing registry from a remote catalog. `url` defaults to
// the blazen.dev Cloudflare Worker, which mirrors models.dev plus live
// OpenRouter / Together pricing on a daily cron.
//
// Returns the number of entries registered. Misses still return `null`
// from `compute_cost`; no automatic retry / cache layer beyond the
// global registry.
func RefreshPricing(url *string) (uint32, error) {
	res, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) C.uint32_t {
			res := C.ffi_blazen_uniffi_rust_future_complete_u32(handle, status)
			return res
		},
		// liftFn
		func(ffi C.uint32_t) uint32 {
			return FfiConverterUint32INSTANCE.Lift(ffi)
		},
		C.uniffi_blazen_uniffi_fn_func_refresh_pricing(FfiConverterOptionalStringINSTANCE.Lower(url)),
		// pollFn
		func(handle C.uint64_t, continuation C.UniffiRustFutureContinuationCallback, data C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_poll_u32(handle, continuation, data)
		},
		// freeFn
		func(handle C.uint64_t) {
			C.ffi_blazen_uniffi_rust_future_free_u32(handle)
		},
	)

	if err == nil {
		return res, nil
	}

	return res, err
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
func NewAnthropicModel(apiKey string, model *string, baseUrl *string) (*Model, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_anthropic_model(FfiConverterStringINSTANCE.Lower(apiKey), FfiConverterOptionalStringINSTANCE.Lower(model), FfiConverterOptionalStringINSTANCE.Lower(baseUrl), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *Model
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build an Azure `OpenAI` chat-completion model.
//
// Azure derives its endpoint from `resource_name` + `deployment_name` and
// its model id from `deployment_name`, so `base_url` is intentionally not
// exposed here. `api_version` defaults to the provider's pinned API
// version when `None`.
func NewAzureModel(apiKey string, resourceName string, deploymentName string, apiVersion *string) (*Model, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_azure_model(FfiConverterStringINSTANCE.Lower(apiKey), FfiConverterStringINSTANCE.Lower(resourceName), FfiConverterStringINSTANCE.Lower(deploymentName), FfiConverterOptionalStringINSTANCE.Lower(apiVersion), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *Model
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build an AWS Bedrock chat-completion model.
//
// `region` selects the AWS region (e.g. `"us-east-1"`); `api_key` is the
// Bedrock API key (which can be obtained via `aws bedrock` IAM keys or
// passed as an empty string to resolve from `AWS_BEARER_TOKEN_BEDROCK`).
func NewBedrockModel(apiKey string, region string, model *string, baseUrl *string) (*Model, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_bedrock_model(FfiConverterStringINSTANCE.Lower(apiKey), FfiConverterStringINSTANCE.Lower(region), FfiConverterOptionalStringINSTANCE.Lower(model), FfiConverterOptionalStringINSTANCE.Lower(baseUrl), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *Model
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterModelINSTANCE.Lift(_uniffiRV), nil
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

// Build a local candle chat-completion model.
//
// Wraps [`CandleLlmProvider`](blazen_llm::CandleLlmProvider) through the
// [`CandleLlmModel`](blazen_llm::CandleLlmModel) trait
// bridge so it satisfies the same `Model` trait as remote
// providers.
func NewCandleModel(modelId string, device *string, quantization *string, revision *string, contextLength *uint32) (*Model, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_candle_model(FfiConverterStringINSTANCE.Lower(modelId), FfiConverterOptionalStringINSTANCE.Lower(device), FfiConverterOptionalStringINSTANCE.Lower(quantization), FfiConverterOptionalStringINSTANCE.Lower(revision), FfiConverterOptionalUint32INSTANCE.Lower(contextLength), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *Model
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build a Cohere chat-completion model.
func NewCohereModel(apiKey string, model *string, baseUrl *string) (*Model, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_cohere_model(FfiConverterStringINSTANCE.Lower(apiKey), FfiConverterOptionalStringINSTANCE.Lower(model), FfiConverterOptionalStringINSTANCE.Lower(baseUrl), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *Model
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Construct a [`Model`] that speaks the `OpenAI` chat-completions
// protocol against an arbitrary base URL.
//
// This is the same wire format as
// [`new_openai_compat_model`], but wrapped in a
// [`blazen_llm::CustomProviderHandle`] for consistent ergonomics with the
// `new_ollama_model` / `new_lm_studio_model`
// factories. `api_key` is optional: passing `None` (or an empty `Some`)
// omits the `Authorization` header entirely.
func NewCustomModelWithOpenaiProtocol(providerId string, baseUrl string, model string, apiKey *string) (*Model, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_custom_model_with_openai_protocol(FfiConverterStringINSTANCE.Lower(providerId), FfiConverterStringINSTANCE.Lower(baseUrl), FfiConverterStringINSTANCE.Lower(model), FfiConverterOptionalStringINSTANCE.Lower(apiKey), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *Model
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build a `DeepSeek` chat-completion model.
func NewDeepseekModel(apiKey string, model *string, baseUrl *string) (*Model, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_deepseek_model(FfiConverterStringINSTANCE.Lower(apiKey), FfiConverterOptionalStringINSTANCE.Lower(model), FfiConverterOptionalStringINSTANCE.Lower(baseUrl), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *Model
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterModelINSTANCE.Lift(_uniffiRV), nil
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

// Build a fal.ai chat-completion model.
//
// `endpoint` selects the fal endpoint family — one of
// `"openai_chat"` (default), `"openai_responses"`, `"openai_embeddings"`,
// `"openrouter"`, `"any_llm"`. Unrecognised values fall back to
// `OpenAiChat`. `enterprise` promotes the endpoint to its enterprise /
// SOC2-eligible variant; `auto_route_modality` toggles automatic routing
// to a vision/audio/video endpoint when the request carries media.
func NewFalModel(apiKey string, model *string, baseUrl *string, endpoint *string, enterprise bool, autoRouteModality bool) (*Model, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_fal_model(FfiConverterStringINSTANCE.Lower(apiKey), FfiConverterOptionalStringINSTANCE.Lower(model), FfiConverterOptionalStringINSTANCE.Lower(baseUrl), FfiConverterOptionalStringINSTANCE.Lower(endpoint), FfiConverterBoolINSTANCE.Lower(enterprise), FfiConverterBoolINSTANCE.Lower(autoRouteModality), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *Model
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterModelINSTANCE.Lift(_uniffiRV), nil
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
func NewFireworksModel(apiKey string, model *string, baseUrl *string) (*Model, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_fireworks_model(FfiConverterStringINSTANCE.Lower(apiKey), FfiConverterOptionalStringINSTANCE.Lower(model), FfiConverterOptionalStringINSTANCE.Lower(baseUrl), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *Model
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build a Google Gemini chat-completion model.
func NewGeminiModel(apiKey string, model *string, baseUrl *string) (*Model, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_gemini_model(FfiConverterStringINSTANCE.Lower(apiKey), FfiConverterOptionalStringINSTANCE.Lower(model), FfiConverterOptionalStringINSTANCE.Lower(baseUrl), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *Model
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build a Groq chat-completion model.
func NewGroqModel(apiKey string, model *string, baseUrl *string) (*Model, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_groq_model(FfiConverterStringINSTANCE.Lower(apiKey), FfiConverterOptionalStringINSTANCE.Lower(model), FfiConverterOptionalStringINSTANCE.Lower(baseUrl), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *Model
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build a local llama.cpp chat-completion model.
//
// `model_path` is either a local GGUF file path or a `HuggingFace` repo
// id; `n_gpu_layers` offloads the given number of layers to the GPU when
// the device supports it.
func NewLlamacppModel(modelPath string, device *string, quantization *string, contextLength *uint32, nGpuLayers *uint32) (*Model, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_llamacpp_model(FfiConverterStringINSTANCE.Lower(modelPath), FfiConverterOptionalStringINSTANCE.Lower(device), FfiConverterOptionalStringINSTANCE.Lower(quantization), FfiConverterOptionalUint32INSTANCE.Lower(contextLength), FfiConverterOptionalUint32INSTANCE.Lower(nGpuLayers), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *Model
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Construct a [`Model`] for an LM Studio server.
//
// Convenience wrapper around [`blazen_llm::lm_studio`] — targets LM Studio's
// local `OpenAI`-compatible endpoint on `http://{host}:{port}/v1`.
func NewLmStudioModel(host string, port uint16, model string) (*Model, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_lm_studio_model(FfiConverterStringINSTANCE.Lower(host), FfiConverterUint16INSTANCE.Lower(port), FfiConverterStringINSTANCE.Lower(model), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *Model
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build a Mistral chat-completion model.
func NewMistralModel(apiKey string, model *string, baseUrl *string) (*Model, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_mistral_model(FfiConverterStringINSTANCE.Lower(apiKey), FfiConverterOptionalStringINSTANCE.Lower(model), FfiConverterOptionalStringINSTANCE.Lower(baseUrl), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *Model
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build a local mistral.rs chat-completion model.
//
// `model_id` is the `HuggingFace` repo id (e.g.
// `"mistralai/Mistral-7B-Instruct-v0.3"`) or a local GGUF path. The
// optional `device`/`quantization` strings follow Blazen's parser format
// (`"cpu"`, `"cuda:0"`, `"metal"`, `"q4_k_m"`, ...). Set `vision = true`
// for multimodal models like LLaVA / Qwen2-VL.
func NewMistralrsModel(modelId string, device *string, quantization *string, contextLength *uint32, vision bool) (*Model, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_mistralrs_model(FfiConverterStringINSTANCE.Lower(modelId), FfiConverterOptionalStringINSTANCE.Lower(device), FfiConverterOptionalStringINSTANCE.Lower(quantization), FfiConverterOptionalUint32INSTANCE.Lower(contextLength), FfiConverterBoolINSTANCE.Lower(vision), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *Model
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Construct a [`Model`] for an Ollama server.
//
// Convenience for [`new_custom_model_with_openai_protocol`] with
// `base_url = format!("http://{host}:{port}/v1")` and no API key. Delegates
// to [`blazen_llm::ollama`], which knows how to speak Ollama's flavour of
// the `OpenAI` chat-completions protocol.
func NewOllamaModel(host string, port uint16, model string) (*Model, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_ollama_model(FfiConverterStringINSTANCE.Lower(host), FfiConverterUint16INSTANCE.Lower(port), FfiConverterStringINSTANCE.Lower(model), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *Model
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build a generic OpenAI-compatible chat-completion model.
//
// Targets any service that speaks the official OpenAI Chat Completions
// wire format (vLLM, llama-server, LM Studio, local proxies, ...). Uses
// `Authorization: Bearer <api_key>` auth.
func NewOpenaiCompatModel(providerName string, baseUrl string, apiKey string, model string) (*Model, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_openai_compat_model(FfiConverterStringINSTANCE.Lower(providerName), FfiConverterStringINSTANCE.Lower(baseUrl), FfiConverterStringINSTANCE.Lower(apiKey), FfiConverterStringINSTANCE.Lower(model), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *Model
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterModelINSTANCE.Lift(_uniffiRV), nil
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

// Build an `OpenAI` chat-completion model.
//
// `base_url` defaults to `https://api.openai.com/v1`; override it to target
// any OpenAI-compatible proxy that uses the official-OpenAI request shape.
func NewOpenaiModel(apiKey string, model *string, baseUrl *string) (*Model, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_openai_model(FfiConverterStringINSTANCE.Lower(apiKey), FfiConverterOptionalStringINSTANCE.Lower(model), FfiConverterOptionalStringINSTANCE.Lower(baseUrl), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *Model
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build an `OpenRouter` chat-completion model.
func NewOpenrouterModel(apiKey string, model *string, baseUrl *string) (*Model, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_openrouter_model(FfiConverterStringINSTANCE.Lower(apiKey), FfiConverterOptionalStringINSTANCE.Lower(model), FfiConverterOptionalStringINSTANCE.Lower(baseUrl), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *Model
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build a Perplexity chat-completion model.
func NewPerplexityModel(apiKey string, model *string, baseUrl *string) (*Model, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_perplexity_model(FfiConverterStringINSTANCE.Lower(apiKey), FfiConverterOptionalStringINSTANCE.Lower(model), FfiConverterOptionalStringINSTANCE.Lower(baseUrl), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *Model
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterModelINSTANCE.Lift(_uniffiRV), nil
	}
}

// Build a Together AI chat-completion model.
func NewTogetherModel(apiKey string, model *string, baseUrl *string) (*Model, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_together_model(FfiConverterStringINSTANCE.Lower(apiKey), FfiConverterOptionalStringINSTANCE.Lower(model), FfiConverterOptionalStringINSTANCE.Lower(baseUrl), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *Model
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterModelINSTANCE.Lift(_uniffiRV), nil
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
func NewXaiModel(apiKey string, model *string, baseUrl *string) (*Model, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_blazen_uniffi_fn_func_new_xai_model(FfiConverterStringINSTANCE.Lower(apiKey), FfiConverterOptionalStringINSTANCE.Lower(model), FfiConverterOptionalStringINSTANCE.Lower(baseUrl), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *Model
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterModelINSTANCE.Lift(_uniffiRV), nil
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
func CompleteStreaming(model *Model, request ModelRequest, sink CompletionStreamSink) error {
	_, err := uniffiRustCallAsync[*BlazenError](
		FfiConverterBlazenErrorINSTANCE,
		// completeFn
		func(handle C.uint64_t, status *C.RustCallStatus) struct{} {
			C.ffi_blazen_uniffi_rust_future_complete_void(handle, status)
			return struct{}{}
		},
		// liftFn
		func(_ struct{}) struct{} { return struct{}{} },
		C.uniffi_blazen_uniffi_fn_func_complete_streaming(FfiConverterModelINSTANCE.Lower(model), FfiConverterModelRequestINSTANCE.Lower(request), FfiConverterCompletionStreamSinkINSTANCE.Lower(sink)),
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
func CompleteStreamingBlocking(model *Model, request ModelRequest, sink CompletionStreamSink) error {
	_, _uniffiErr := rustCallWithError[*BlazenError](FfiConverterBlazenError{}, func(_uniffiStatus *C.RustCallStatus) bool {
		C.uniffi_blazen_uniffi_fn_func_complete_streaming_blocking(FfiConverterModelINSTANCE.Lower(model), FfiConverterModelRequestINSTANCE.Lower(request), FfiConverterCompletionStreamSinkINSTANCE.Lower(sink), _uniffiStatus)
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
