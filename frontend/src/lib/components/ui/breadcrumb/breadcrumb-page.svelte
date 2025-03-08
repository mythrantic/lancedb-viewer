<script>
	import { cn } from '$lib/utils.js';

	let { ref = $bindable(null), class: className, children, ...restProps } = $props();
</script>

<span
	bind:this={ref}
	role="link"
	aria-disabled="true"
	aria-current="page"
	class={cn('font-normal text-foreground', className)}
	{...restProps}
>
	{@render children?.()}
</span>
