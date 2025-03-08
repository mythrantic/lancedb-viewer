<script>
	import * as Sidebar from '$lib/components/ui/sidebar/index.js';
	let { ref = $bindable(null), items, ...restProps } = $props();
</script>

<Sidebar.Group bind:ref {...restProps}>
	<Sidebar.GroupContent>
		<Sidebar.Menu>
			{#each items as item (item.title)}
				<Sidebar.MenuItem>
					<Sidebar.MenuButton size="sm">
						{#snippet child({ props })}
							<a href={item.url} {...props}>
								<item.icon />
								<span>{item.title}</span>
							</a>
						{/snippet}
					</Sidebar.MenuButton>
				</Sidebar.MenuItem>
			{/each}
		</Sidebar.Menu>
	</Sidebar.GroupContent>
</Sidebar.Group>
