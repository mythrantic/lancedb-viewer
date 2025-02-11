<script>
	import { cn } from "$lib/utils.js";

	let {
		ref = $bindable(null),
		class: className,
		href = undefined,
		child,
		children,
		...restProps
	} = $props();

	const attrs = $derived({
		class: cn("hover:text-foreground transition-colors", className),
		href,
		...restProps,
	});
</script>

{#if child}
	{@render child({ props: attrs })}
{:else}
	<a bind:this={ref} {...attrs}>
		{@render children?.()}
	</a>
{/if}
