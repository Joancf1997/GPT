<template>
    <div class="w-screen h-screen flex flex-col text-white cuerpo">
      <div class="">
        <ModelSelection />
      </div>
      <div class="chatSection"> 
        <Chat/>
      </div>
      
    </div>
  </template>
  
  <script setup>
  import { ref } from "vue";
  import ModelSelection from './ModelSelection.vue';
  import Chat from './Chat.vue';
  
  const messages = ref(["Welcome to the chat!", "How can I help you?"]);
  const newMessage = ref("");
  
  const sendMessage = () => {
    if (newMessage.value.trim() !== "") {
      messages.value.push(newMessage.value);
      newMessage.value = "";
    }
  };
  </script>
  
<style scoped>
.cuerpo{ 
    width: 100vw;
    height: 100vh;
    padding:0px;
    border:2px solid red;
}

.caja{ 
    border:2px solid green
}


.chatSection{ 
    border:2px solid green
}


</style>