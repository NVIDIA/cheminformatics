<template>
  <div style="height: 100vh; width: 100vw">
    <baklava-editor :plugin="viewPlugin" />
  </div>
</template>

<script>
import Vue from 'vue'
import { Editor, Node } from '@baklavajs/core'
import { ViewPlugin } from '@baklavajs/plugin-renderer-vue'
import { OptionPlugin } from '@baklavajs/plugin-options-vue'
import { Engine } from '@baklavajs/plugin-engine'

import PrepareProtineSideBar from '../../components/sidebar/PrepareProtineSideBar'

export class PrepareProtine extends Node {
  constructor () {
    super()
    this.type = 'PrepareProtine'
    this.name = 'Prepare Protine'

    this.addInputInterface('protein-file', 'InputOption')
    this.addOption('Command',
      'ButtonOption',
      'bash',
      PrepareProtineSideBar.name)
    this.addOutputInterface('protein-receptor-file')
  }
}

export class PrepareLigand extends Node {
  constructor () {
    super()
    this.type = 'PrepareLigand'
    this.name = 'Prepare Ligand'

    this.addInputInterface('SMILES', 'InputOption')
    this.addOutputInterface('ligands-dir')
    this.addOutputInterface('library-file')
  }
}

export class GeneratePose extends Node {
  constructor () {
    super()
    this.type = 'GeneratePose'
    this.name = 'Generate Pose'

    this.addInputInterface('protein-receptor-file', 'InputOption')
    this.addInputInterface('ligands-dir', 'InputOption')

    this.addOutputInterface('docked-dir')
  }
}

export class ScorePose extends Node {
  constructor () {
    super()
    this.type = 'ScorePose'
    this.name = 'Score Pose'

    this.addInputInterface('library-file', 'InputOption')
    this.addInputInterface('docked-dir', 'InputOption')

    this.addOutputInterface('score-stats')
    this.addOutputInterface('score-file')
  }
}

export class GenerateReport extends Node {
  constructor () {
    super()
    this.type = 'GenerateReport'
    this.name = 'Generate Report'

    this.addInputInterface('score-file', 'InputOption')
    this.addOutputInterface('viz-result')
  }
}

export default Vue.extend({
  name: 'Workflow',
  components: { },
  data () {
    return {
      editor: new Editor(),
      viewPlugin: new ViewPlugin(),
      engine: new Engine(true),
      value: ''
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

    // this.addNodeWithCoordinates(PrepareProtine, 100, 140)
    // this.addNodeWithCoordinates(PrepareLigand, 400, 140)
    // this.addNodeWithCoordinates(GeneratePose, 700, 140)
    // this.addNodeWithCoordinates(ScorePose, 1000, 140)
    // this.addNodeWithCoordinates(GenerateReport, 100, 440)
    // this.editor.addConnection(
    //   node1.getInterface("Result"),
    //   node2.getInterface("Value")
    // );
    this.engine.calculate()
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
