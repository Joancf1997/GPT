<template>
  <div class="w-screen h-screen flex flex-col text-white cuerpo">
    <div class="menuSection"> 
      <ModelSelection  @model-selected="handleModelSelection"/>
    </div>
    <div class="chatSection"> 
      <Chat :selectedModel="selectedModel"/>
    </div>
    
  </div>
</template>

<script setup>
import { ref } from "vue";
import ModelSelection from './components/ModelSelection.vue';
import Chat from './components/Chat.vue';

const selectedModel = ref("");


const handleModelSelection = (model) => {
  selectedModel.value = model;
};
</script>

<style scoped>
.cuerpo {
  padding: 10px;
  display: flex;
  flex-direction: column;
  width: 100vw;
  height: 100vh; 
  padding-bottom:10vh;
  background-color: #f5f5f5;

}


.menuSection{
  min-height: 55px;
  width: 100%;
}

.chatSection{ 
  flex-grow: 1; 
  width: 50%;
  margin-left: 25%;
}


</style>