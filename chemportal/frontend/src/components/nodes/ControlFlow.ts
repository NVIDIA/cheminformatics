import { Node } from '@baklavajs/core'
import LoopNewInput from './sidebar/LoopNewInput.vue'
import LoopExpression from './sidebar/LoopExpression.vue'

export class Loop extends Node {
  public type = 'Loop'
  public name = this.type
  public expression = '';

  constructor () {
    super()

    this.addOption('Expression',
      'ButtonOption',
      { LoopNode: this },
      LoopExpression.name)

    this.addOption('Add Input',
      'ButtonOption',
      { LoopNode: this },
      LoopNewInput.name)

    this.addInputInterface('From Loop', 'InputOption')

    this.addOutputInterface('To Exit')
  }
}
