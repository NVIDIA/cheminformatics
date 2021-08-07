export default {

  methods: {
    resetBusy () {
      this.busy = false
    },

    showMsg (obj, msg, type) {
      this.$store.state.message.show = true
      this.$store.state.message.text = msg
      this.$store.state.message.level = type

      this.resetBusy()
    },

    showError (obj, error) {
      let errorMsg = 'Unexpected error'
      // Best effort to get error message
      if (error !== undefined) errorMsg = error
      if (
        error.response !== undefined &&
        error.response.data !== undefined &&
        error.response.data.error_msg !== undefined
      ) {
        errorMsg = error.response.data.error_msg
      }

      this.showMsg(obj, errorMsg, 'error')
      this.resetBusy()
    }
  }
}
