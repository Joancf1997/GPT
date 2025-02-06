<template>
    <!-- Chat Input Section -->
    <div class="chatBody">
        <!-- Chat Messages Section -->
        <div class="flex-grow overflow-y-auto chatSection">
					<ScrollPanel ref="scrollPanelRef" style="width: 100%; height: 80vh;">
            <div v-for="(msg, index) in messages" :key="index" class="mb-3">
							<Card v-if="msg.user" class="mt-2">
								<template #title> 
									<Avatar image="/Logos/User.webp" class="mr-2" size="xlarge" shape="circle" />
								</template>
								<template #content>
									<p class="m-0">
										{{ msg.text }}
									</p>
								</template>
							</Card>
							<Card v-if="!msg.user" class="chat-card mt-2">
								<template #title v-if="!msg.user" > 
									<div style="display: flex; justify-content: flex-end; align-items: center;">
										<h4 class="m-2" v-if="msg.model == 'Assistant'"> Assistant Model </h4>
										<h4 class="m-2" v-if="msg.model == 'classification'"> Classification Model </h4>
										<Avatar v-if="msg.model == 'Assistant'" image="/Logos/GPT-Logo.webp" class="mr-2" size="xlarge" shape="circle" />
										<Avatar v-if="msg.model == 'classification'" image="/Logos/GPT-white-Logo.webp" class="mr-4" size="xlarge" shape="circle" />
									</div>
								</template>
								<template #content>
									<div style="display: flex; justify-content: flex-end; align-items: center;">

										<p class="m-0">
												{{ msg.text }}
										</p>
									</div>
								</template>
							</card>
            </div>
					</ScrollPanel>
        </div>
        <div class="flex items-center space-x-2 inputSection">
					<InputText v-if="currentModel == 'Assistant'" v-model="modelPrompt" placeholder="Instruction.." class="flex-grow text-black inputText"/>
					<InputText v-model="newMessage" :placeholder="label" class="flex-grow text-black inputText"/>
					<Button icon="pi pi-arrow-up" aria-label="Save" @click="sendMessage" rounded severity="contrast" class="m-3"/>
        </div>
    </div>
</template>


<script setup>
  import { defineProps, watch, ref } from "vue";
	import Card from 'primevue/card';
	import ScrollPanel from 'primevue/scrollpanel';
	import Avatar from 'primevue/avatar';
	import axios from "axios";

	const props = defineProps({
		selectedModel: String,
	});
	const currentModel = ref(props.selectedModel);
	const label = ref("")

	const scrollPanelRef = ref(null);
  const messages = ref([]);
  const newMessage = ref("");
  const modelPrompt = ref("");


	watch(() => props.selectedModel, (newValue, ) => {
		currentModel.value = newValue
		if(newValue == 'Assistant'){ 
			label.value = "Type your input..."
		} else {
			label.value = "Type your message..."
		}
	});




  const sendMessage = async () => {
		if(currentModel.value == 'Assistant'){ 
			assistantModel();
		} else { 
			classificationModel();
		}
		newMessage.value = ""
	};



	const classificationModel = async() => { 
		messages.value.push({
			text: newMessage.value,
			user: true,
			model: 'classification'
		});
		try {
			const response = await axios.post("http://127.0.0.1:4000/ClassificationMsg", {
				input: newMessage.value
			});
			let model_response = response.data.response;
			// Model response
			messages.value.push({
				text: model_response,
				user: false,
				model: 'classification'
			});
		} catch (error) {
			console.log("Error on the response from the model");
			console.log(error);
		}
	}




	const assistantModel = async() => { 
		messages.value.push({
			text: modelPrompt.value != "" ? modelPrompt.value + ": " + newMessage.value : newMessage.value,
			user: true,
			model: 'Assistant'
		});
		try {
			const response = await axios.post("http://127.0.0.1:4000/AssistantMsg", {
				instruction: modelPrompt.value,
				input: newMessage.value
			});
			let model_response = response.data.response;
			// Model response
			messages.value.push({
				text: model_response,
				user: false,
				model: 'Assistant'
			});
		} catch (error) {
			console.log("Error on the response from the model");
			console.log(error);
		}
	}


</script>


<style scoped>

.btnInput{
    width: 15%;
}

.inputText{
    width: 70%;
    margin-left: 11%;
}

.chatBody{
    width: 100%;
		padding-top: 5%;
    height: 100%;
    display: flex;
    flex-direction: column;
}

.chatSection{ 
  flex-grow: 1; 
  width: 100%;
}

.inputSection{
  height: 55px;
	margin: left 7%;
  width: 100%;
}

.user-card {
    background-color: #f5f5f5;
    padding: 1rem;
    border-radius: 8px;
}

.chat-card {
    background-color: #f5f5f5;
    padding: 1rem;
    border-radius: 8px;
}
</style>
