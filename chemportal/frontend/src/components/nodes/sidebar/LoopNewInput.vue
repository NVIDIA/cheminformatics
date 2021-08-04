<template>
    <v-card elevation="2">
        <v-card-title class="primary">New Input</v-card-title>
        <v-card-text>
            <v-text-field
                v-model="inputName"
                label="Input Name"
                placeholder="Input Name">
            </v-text-field>
            <v-btn @click="addInput" class="primary white--text">
                Add
            </v-btn>
        </v-card-text>
    </v-card>
</template>

<script lang="ts">
import { Component, Prop, Vue, Inject } from 'vue-property-decorator'
import { ViewPlugin } from '@baklavajs/plugin-renderer-vue'

@Component
export default class LoopNewInput extends Vue {
  @Prop({ type: Object })
  public value

  @Inject('plugin')
  plugin!: ViewPlugin

  data () {
    return {
      inputName: ''
    }
  }

  addInput () {
    if (this.inputName === '' || this.inputName === undefined) {
      return
    }
    this.value.LoopNode.addInputInterface('Input ' + this.inputName, 'InputOption')
    this.value.LoopNode.addOutputInterface('Output ' + this.inputName)
    this.plugin.sidebar.visible = false
  }
}
</script>
