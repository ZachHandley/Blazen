<script setup lang="ts">
import { ref } from "vue";

interface Tab {
  label: string;
  language: string;
  code: string;
}

interface Props {
  tabs: Tab[];
}

const props = defineProps<Props>();

const activeIndex = ref(0);

function setActive(index: number): void {
  activeIndex.value = index;
}
</script>

<template>
  <div class="overflow-hidden rounded-xl border border-surface-600">
    <div class="flex border-b border-surface-600 bg-surface-800">
      <button
        v-for="(tab, index) in props.tabs"
        :key="tab.label"
        class="relative px-5 py-3 text-sm font-medium transition-colors duration-200"
        :class="
          activeIndex === index
            ? 'text-blazen-400'
            : 'text-gray-400 hover:text-gray-200'
        "
        @click="setActive(index)"
      >
        {{ tab.label }}
        <span
          v-if="activeIndex === index"
          class="absolute bottom-0 left-0 h-0.5 w-full bg-blazen-500"
        />
      </button>
    </div>
    <div class="bg-[var(--color-code-bg)] p-5">
      <pre
        class="overflow-x-auto text-sm leading-relaxed text-gray-300"
      ><code>{{ props.tabs[activeIndex].code }}</code></pre>
    </div>
  </div>
</template>
