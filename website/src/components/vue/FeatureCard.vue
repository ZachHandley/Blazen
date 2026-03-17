<script setup lang="ts">
interface Props {
  icon: string;
  title: string;
  description: string;
}

defineProps<Props>();
</script>

<template>
  <div
    class="group relative rounded-xl border border-surface-600 bg-surface-800 p-6 transition-all duration-300 hover:border-blazen-500/40 hover:bg-surface-700"
  >
    <div
      class="mb-4 flex h-10 w-10 items-center justify-center rounded-lg bg-blazen-500/10 text-2xl"
    >
      {{ icon }}
    </div>
    <h3 class="mb-2 text-lg font-semibold text-gray-100">{{ title }}</h3>
    <p class="text-sm leading-relaxed text-gray-400">{{ description }}</p>
  </div>
</template>
