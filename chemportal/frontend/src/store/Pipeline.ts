import axios from 'axios'

export default {
  namespaced: true,
  state: {
    list: [],
    pipeline: null
  },

  mutations: {
    fetchAll (state, args) {
      state.list = []

      axios.get('./api/pipeline').then(response => {
        state.list = response.data
        if (args.onSuccess) {
          args.onSuccess(state.list)
        }
      }).catch(function (error) {
        if (args.onFailure) {
          args.onFailure(null, error)
        }
      })
    },

    save (state, args) {
      console.log(args)
      axios.post('./api/pipeline', args.pipeline).then(response => {
        if (args.onSuccess) {
          args.onSuccess(args.pipeline)
        }
      }).catch(function (error) {
        if (args.onFailure) args.onFailure(args.pipeline, error)
      })
    }
  }
}
