import Vue from 'vue'
import Vuex from 'vuex'
import pipeline from './Pipeline'
import message from './Message'

Vue.use(Vuex)

export default new Vuex.Store({
  modules: {
    pipeline,
    message
  }
})
