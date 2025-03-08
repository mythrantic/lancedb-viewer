<script>
	import { Separator } from '$lib/components/ui/separator/index.js';
	import { cn } from '$lib/utils.js';
	let { ref = $bindable(null), class: className, ...restProps } = $props();
</script>

<Separator
	bind:ref
	data-sidebar="separator"
	class={cn('mx-2 w-auto bg-sidebar-border', className)}
	{...restProps}
/>
