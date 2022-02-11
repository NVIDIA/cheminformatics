
export default {
  namespaced: true,
  state: {
    show: false,
    text: '',
    level: 'info'
  },

  mutations: {
    showMessage (state, args) {
      state.settings.text = args.msg
      if (args.msgLevel === null || args.msgLevel === undefined) {
        state.settings.level = 'info'
      } else {
        state.settings.level = args.msgLevel
      }
      state.settings.show = true
    }
  }
}
