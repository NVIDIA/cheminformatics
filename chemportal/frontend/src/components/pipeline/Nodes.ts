import { Node } from '@baklavajs/core'
import PrepareProtineSideBar from './PrepareProtineSideBar.vue'

export class PrepareProtine extends Node {
  public type = 'PrepareProtine'
  public name = this.type;

  constructor () {
    super()

    this.addInputInterface('protein-file', 'InputOption')
    this.addOption('Command',
      'ButtonOption',
      'bash',
      PrepareProtineSideBar.name)
    this.addOutputInterface('protein-receptor-file')
  }
}

export class PrepareLigand extends Node {
  public type = 'PrepareLigand'
  public name = this.type;

  constructor () {
    super()

    this.addInputInterface('SMILES', 'InputOption')
    this.addOutputInterface('ligands-dir')
    this.addOutputInterface('library-file')
  }
}

export class GeneratePose extends Node {
  public type = 'GeneratePose'
  public name = this.type;

  constructor () {
    super()

    this.addInputInterface('protein-receptor-file', 'InputOption')
    this.addInputInterface('ligands-dir', 'InputOption')

    this.addOutputInterface('docked-dir')
  }
}

export class ScorePose extends Node {
  public type = 'ScorePose'
  public name = this.type;

  constructor () {
    super()

    this.addInputInterface('library-file', 'InputOption')
    this.addInputInterface('docked-dir', 'InputOption')

    this.addOutputInterface('score-stats')
    this.addOutputInterface('score-file')
  }
}

export class GenerateReport extends Node {
  public type = 'GenerateReport'
  public name = this.type;

  constructor () {
    super()

    this.addInputInterface('score-file', 'InputOption')
    this.addOutputInterface('viz-result')
  }
}
