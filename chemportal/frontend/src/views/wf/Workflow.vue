<template>
  <v-layout column fill-height text-xs-center>
    <v-card>
      <v-toolbar cards flat>
        <v-card-title>
          Pipeline
        </v-card-title>
        <v-spacer></v-spacer>
        <v-btn
          class='ma-2'
          @click="$router.push('/wf/list')">
          Cancel
        </v-btn>

        <v-btn
          :disabled="!formValid"
          class="primary ma-2 white--text"
          @click='save()'>
          Save
          <v-icon right dark>mdi-delete</v-icon>
        </v-btn>
      </v-toolbar>

      <v-form
        ref="form"
        v-model="formValid"
        class="pa-4 pt-6">

        <v-text-field
          v-model="pipeline.name"
          :rules="rules.required"
          label="Name">
        </v-text-field>
        <v-textarea
          v-model="pipeline.description"
          :rules="rules.required"
          auto-grow
          label="Description"
          rows="2">
        </v-textarea>
      </v-form>
    </v-card>

    <baklava-editor :plugin="viewPlugin" ondrop='alert("test");'/>
  </v-layout>
</template>

<script>
import Vue from 'vue'
import { Editor } from '@baklavajs/core'
import { ViewPlugin } from '@baklavajs/plugin-renderer-vue'
import { OptionPlugin } from '@baklavajs/plugin-options-vue'
import { Engine } from '@baklavajs/plugin-engine'

import { PrepareProtine, PrepareLigand, GeneratePose, ScorePose, GenerateReport, GenerateMolecules }
  from '../../components/nodes/Domain.ts'
import { Loop } from '../../components/nodes/ControlFlow.ts'
import PrepareProtineSideBar from '../../components/nodes/sidebar/PrepareProtineSideBar'
import LoopNewInput from '../../components/nodes/sidebar/LoopNewInput.vue'
import LoopExpression from '../../components/nodes/sidebar/LoopExpression.vue'

import CommonView from '../../mixin/CommonView'

export default Vue.extend({
  name: 'Workflow',
  components: { },
  mixins: [CommonView],
  data () {
    return {
      editor: new Editor(),
      viewPlugin: new ViewPlugin(),
      engine: new Engine(true),
      formValid: false,
      pipeline: {
        name: null,
        description: null
      },
      editMode: false,
      rules: {
        email: [v => !!(v || '').match(/@/) || 'Please enter a valid email'],
        length: [len => v => (v || '').length >= len || `Invalid character length, required ${len}`],
        password: [v => !!(v || '').match(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*(_|[^\w])).+$/) ||
          'Password must contain an upper case letter, a numeric character, and a special character'],
        required: [v => !!v || 'Required value']
      }
    }
  },
  created () {
    this.editor.use(this.viewPlugin)
    this.editor.use(new OptionPlugin())
    this.editor.use(this.engine)

    this.viewPlugin.enableMinimap = false
    this.viewPlugin.registerOption('PrepareProtineSideBar', PrepareProtineSideBar)
    this.viewPlugin.registerOption('LoopNewInput', LoopNewInput)
    this.viewPlugin.registerOption('LoopExpression', LoopExpression)

    this.editor.registerNodeType('GenerateMolecules', GenerateMolecules)
    this.editor.registerNodeType('PrepareProtine', PrepareProtine)
    this.editor.registerNodeType('PrepareLigand', PrepareLigand)
    this.editor.registerNodeType('GeneratePose', GeneratePose)
    this.editor.registerNodeType('ScorePose', ScorePose)
    this.editor.registerNodeType('GenerateReport', GenerateReport)
    this.editor.registerNodeType('Loop', Loop)
  },

  mounted () {
    var pipelineId = this.$route.params.id
    if (pipelineId !== undefined) {
      this.editMode = true
      this.fetch(pipelineId)
    }
  },

  methods: {
    fetch (pipelineId) {
      this.$store.dispatch('pipeline/fetch', pipelineId).then(response => {
        this.pipeline = response.data.data
      }).catch(error => {
        console.log(error)
        this.showError(pipelineId, error)
      })
    },

    save () {
      this.$store.dispatch('pipeline/save', this.pipeline).then(response => {
        this.showMsg(this.pipeline, 'Pipeline "' + this.pipeline.name + '" saved ', 'info')
        this.$router.push('/wf/list')
      }).catch(error => {
        console.log(error)
        this.showError(this.pipeline, error)
      })
    },

    addNodeWithCoordinates (NodeType, x, y) {
      const n = new NodeType()
      this.editor.addNode(n)
      n.position.x = x
      n.position.y = y
      return n
    }
  }
})
</script>
