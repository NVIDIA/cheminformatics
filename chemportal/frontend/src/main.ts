import Vue from 'vue'
import App from './App.vue'
import router from './router'
import store from './store'

import { BaklavaVuePlugin } from '@baklavajs/plugin-renderer-vue'
import '@baklavajs/plugin-renderer-vue/dist/styles.css'
import vuetify from './plugins/vuetify'

Vue.use(BaklavaVuePlugin)

Vue.config.productionTip = false

new Vue({
  router,
  store,
  vuetify,
  render: h => h(App)
}).$mount('#app')
