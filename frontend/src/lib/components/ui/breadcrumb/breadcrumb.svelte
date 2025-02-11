<script>
	let { ref = $bindable(), class: className, children, ...restProps } = $props();
</script>

<nav bind:this={ref} class={className} aria-label="breadcrumb" {...restProps}>
	{@render children?.()}
</nav>
