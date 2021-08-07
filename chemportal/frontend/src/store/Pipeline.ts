import axios from 'axios'

export default {
  namespaced: true,
  state: {
    list: []
  },

  mutations: {
    add (state, pipeline) {
      state.list.unshift(pipeline)
    },

    refreshLists (state, pipelines) {
      state.list = pipelines.data
    }
  },

  actions: {
    async save ({ commit }, pipeline) {
      let response = null

      if (pipeline.id === undefined) {
        // With no pipelineId the request is a POST
        response = await axios.post('./api/pipeline', pipeline)
        pipeline.id = response.data.data
        commit('add', pipeline)
      } else {
        response = await axios.patch('./api/pipeline', pipeline)
        console.log(response)
      }

      return response
    },

    async fetchAll ({ commit, state }, force) {
      if (force || state.list.length === 0) {
        const response = await axios.get('./api/pipelines/0/10')
        commit('refreshLists', response.data)
        return response
      }
    },

    async fetch ({ commit }, pipelineId) {
      return await axios.get('./api/pipeline/' + pipelineId)
    }
  }
}
