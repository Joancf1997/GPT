<template>
    <div class="card">
        <Menubar :model="items" />
    </div>
</template>

<script setup>
import { ref } from 'vue';
import Menubar from 'primevue/menubar';


const emit = defineEmits(["model-selected"]);
const items = ref([
	{
		label: 'Classification',
		icon: 'pi pi-microchip-ai',
		command: () => onSelect("Classification"),
	},
	{
		label: 'Assistant',
		icon: 'pi pi-microchip-ai',
		command: () => onSelect("Assistant"),
	}
]);


const onSelect = (item) => { 
	emit("model-selected", item);
}




</script>

<style scoped>
.card{
    width: 100%;
    height: 55px;
}
</style>
