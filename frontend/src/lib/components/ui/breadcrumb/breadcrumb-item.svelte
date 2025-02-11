<script>
	import { cn } from "$lib/utils.js";

	let { ref = $bindable(null), class: className, children, ...restProps } = $props();
</script>

<li bind:this={ref} class={cn("inline-flex items-center gap-1.5", className)} {...restProps}>
	{@render children?.()}
</li>
