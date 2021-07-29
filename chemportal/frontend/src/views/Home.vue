<template>
  <Dashboard />
</template>

<script lang="ts">
import Vue from 'vue'
import axios from 'axios'

import Dashboard from '../components/Dashboard.vue'

export default Vue.extend({
  name: 'Home',

  components: {
    Dashboard
  },

  mounted () {
    axios.get('/api/workflow')
      .then(response => {
        console.log(response.data)
      })
      .catch(function (error) {
        console.log(error)
      })
  }

})
</script>
