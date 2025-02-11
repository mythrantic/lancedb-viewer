<script>
	import * as Resizable from '$lib/components/ui/resizable/index.js';
	import { toast } from 'svelte-sonner';
	import { Button } from '$lib/components/ui/button/index.js';
</script>

<Resizable.PaneGroup direction="horizontal" class="rounded-lg border">
	<Resizable.Pane defaultSize={500}>
		<Button
			variant="outline"
			onclick={() =>
				toast.success('Event has been created', {
					description: 'Sunday, December 03, 2023 at 9:00 AM',
					action: {
						label: 'Undo',
						onClick: () => console.info('Undo')
					}
				})}
		>
			Show Toast
		</Button>
	</Resizable.Pane>
	<Resizable.Handle />
	<Resizable.Pane defaultSize={500}>
		<Resizable.PaneGroup direction="vertical">
			<Resizable.Pane defaultSize={250}>
				<div class="flex h-full items-center justify-center p-6">
					<span class="font-semibold">Two</span>
				</div>
			</Resizable.Pane>
			<Resizable.Handle />
			<Resizable.Pane defaultSize={750}>
				<div class="flex h-full items-center justify-center p-6">
					<span class="font-semibold">Three</span>
				</div>
			</Resizable.Pane>
		</Resizable.PaneGroup>
	</Resizable.Pane>
</Resizable.PaneGroup>
