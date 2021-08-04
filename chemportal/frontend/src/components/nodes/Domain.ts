import { Node } from '@baklavajs/core'

export class GenerateMolecules extends Node {
  public type = 'GenerateMolecules'
  public name = this.type;

  constructor () {
    super()

    this.addInputInterface('smiles', 'InputOption')
    this.addOption('Generative Model', 'SelectOption', 'MegaMolBART', undefined,
      { items: ['MegaMolBART', 'CDDD'] })

    this.addOutputInterface('generated-smiles')
  }
}

export class PrepareProtine extends Node {
  public type = 'PrepareProtine'
  public name = this.type;

  constructor () {
    super()

    this.addInputInterface('protein-file', 'InputOption')

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

    this.addInputInterface('docked-dir', 'InputOption')
    this.addInputInterface('library-file', 'InputOption')

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
