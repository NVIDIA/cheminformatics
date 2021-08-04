<template>
    <v-card elevation="2">
        <v-card-title class="primary">Loop Expression</v-card-title>
        <v-card-text>
            <v-textarea
                v-model="expression"
                label="Expression"
                :rows="10">
            </v-textarea>
            <v-btn @click="setExpression" class="primary white--text">
                Ok
            </v-btn>
        </v-card-text>
    </v-card>
</template>

<script lang="ts">
import { Component, Prop, Vue, Inject } from 'vue-property-decorator'
import { ViewPlugin } from '@baklavajs/plugin-renderer-vue'

@Component
export default class LoopExpression extends Vue {
  @Prop({ type: Object })
  public value

  @Inject('plugin')
  plugin!: ViewPlugin

  data () {
    return {
      expression: ''
    }
  }

  mounted () {
    this.$nextTick(function () {
      this.expression = this.value.LoopNode.expression
    })
  }

  setExpression () {
    if (this.expression === '' || this.expression === undefined) {
      return
    }
    this.value.LoopNode.expression = this.expression
    this.plugin.sidebar.visible = false
  }
}
</script>
