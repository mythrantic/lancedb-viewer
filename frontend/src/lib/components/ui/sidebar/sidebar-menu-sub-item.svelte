<script>
	let { ref = $bindable(null), children, ...restProps } = $props();
</script>

<li bind:this={ref} data-sidebar="menu-sub-item" {...restProps}>
	{@render children?.()}
</li>
