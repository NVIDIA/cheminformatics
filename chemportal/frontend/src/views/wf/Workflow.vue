<template>
  <v-layout column fill-height text-xs-center>
    <v-card>
      <v-toolbar cards flat>
        <v-card-title>
          Pipeline
        </v-card-title>
        <v-spacer></v-spacer>
        <v-btn style="margin-right: 6px;">Cancel</v-btn>

        <v-btn
          color="red"
          class="primary ma-2 white--text"
        >
          Save
          <v-icon right dark>mdi-delete</v-icon>
        </v-btn>

      </v-toolbar>

      <v-form
        ref="form"
        v-model="form"
        class="pa-4 pt-6">

        <v-text-field
          v-model="name"
          filled
          label="Name"
        ></v-text-field>
        <v-textarea
          v-model="description"
          auto-grow
          filled
          label="Description"
          rows="2"
        ></v-textarea>

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

import { PrepareProtine, PrepareLigand, GeneratePose, ScorePose, GenerateReport } from '../../components/pipeline/Nodes.ts'
import PrepareProtineSideBar from '../../components/pipeline/PrepareProtineSideBar'

export default Vue.extend({
  name: 'Workflow',
  components: { },
  data () {
    return {
      editor: new Editor(),
      viewPlugin: new ViewPlugin(),
      engine: new Engine(true),
      value: '',
      rules: {
        email: v => !!(v || '').match(/@/) || 'Please enter a valid email',
        length: len => v => (v || '').length >= len || `Invalid character length, required ${len}`,
        password: v => !!(v || '').match(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*(_|[^\w])).+$/) ||
          'Password must contain an upper case letter, a numeric character, and a special character',
        required: v => !!v || 'This field is required'
      }
    }
  },
  created () {
    this.editor.use(this.viewPlugin)
    this.editor.use(new OptionPlugin())
    this.editor.use(this.engine)

    this.viewPlugin.enableMinimap = false
    this.viewPlugin.registerOption('PrepareProtineSideBar', PrepareProtineSideBar)

    this.editor.registerNodeType('PrepareProtine', PrepareProtine)
    this.editor.registerNodeType('PrepareLigand', PrepareLigand)
    this.editor.registerNodeType('GeneratePose', GeneratePose)
    this.editor.registerNodeType('ScorePose', ScorePose)
    this.editor.registerNodeType('GenerateReport', GenerateReport)
  },

  methods: {

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
