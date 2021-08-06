export default {

  methods: {
    resetBusy () {
      this.busy = false
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

      this.$store.state.message.show = true
      this.$store.state.message.text = errorMsg
      this.$store.state.message.level = 'error'

      this.resetBusy()
    }
  }
}
