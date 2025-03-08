<script>
	import { cn } from '$lib/utils.js';

	let { ref = $bindable(null), class: className, ...restProps } = $props();
</script>

<div
	bind:this={ref}
	class={cn('animate-pulse rounded-md bg-muted', className)}
	{...restProps}
></div>
