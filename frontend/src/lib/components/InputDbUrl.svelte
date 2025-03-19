<script lang="ts">
    import { Button } from "$lib/components/ui/button/index.js";
    import { Input } from "$lib/components/ui/input/index.js";
</script>

<form class="flex flex-col w-full  items-center space-y-2">
    <Input type="text" placeholder="input lancedb local url absolute path" />
    <Button type="submit" class="w-2/3">Load Database</Button>
</form>