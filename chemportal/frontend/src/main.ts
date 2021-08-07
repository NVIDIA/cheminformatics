import Vue from 'vue'
import App from './App.vue'
import router from './router'
import store from './store'

import { BaklavaVuePlugin } from '@baklavajs/plugin-renderer-vue'
import '@baklavajs/plugin-renderer-vue/dist/styles.css'
import vuetify from './plugins/vuetify'
import moment from 'moment-timezone'

Vue.use(BaklavaVuePlugin)

Vue.config.productionTip = false

Vue.prototype.$moment = moment

Vue.filter('truncate', function (text, length, suffix) {
  if (text.length > length) {
    return text.substring(0, length) + suffix
  } else {
    return text
  }
})

new Vue({
  router,
  store,
  vuetify,
  render: h => h(App)
}).$mount('#app')
