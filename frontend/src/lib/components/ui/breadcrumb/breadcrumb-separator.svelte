<script>
	import ChevronRight from "lucide-svelte/icons/chevron-right";
	import { cn } from "$lib/utils.js";

	let { ref = $bindable(null), class: className, children, ...restProps } = $props();
</script>

<li
	role="presentation"
	aria-hidden="true"
	class={cn("[&>svg]:size-3.5", className)}
	bind:this={ref}
	{...restProps}
>
	{#if children}
		{@render children?.()}
	{:else}
		<ChevronRight />
	{/if}
</li>
