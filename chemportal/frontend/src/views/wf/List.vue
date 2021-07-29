<template>
  <v-container>
    <v-toolbar
      flat
      style="background: inherit; margin-top: 24px; vertical-align: text-top"
    >
      <v-toolbar-title>Workflows</v-toolbar-title>
      <v-btn icon>
        <v-icon>refresh</v-icon>
      </v-btn>

      <v-btn class="ma-2"
        @click="$router.push('/wf/workflow')">
        New
        <v-icon right dark>add_circle</v-icon>
      </v-btn>

      <v-btn
        :disabled="selected.length == 0"
        color="red"
        class="ma-2 white--text"
      >
        Delete
        <v-icon right dark>mdi-delete</v-icon>
      </v-btn>

      <v-spacer></v-spacer>
      <v-text-field
        :full-width="true"
        v-model="wildSearch"
        prepend-icon="search"
      ></v-text-field>
    </v-toolbar>

    <v-data-table
      :headers="wfHeaders"
      item-key="id"
      :search="wildSearch"
      show-select
      :options="options"
      :sort-desc="true"
      class="elevation-1"
    >
    </v-data-table>
  </v-container>
</template>

<script>
import Vue from 'vue'

export default Vue.extend({
  name: 'WfList',
  components: {
    //
  },
  data: function () {
    return {
      wildSearch: '',
      selected: [],
      options: {
        itemsPerPage: 20
      },
      wfHeaders: [
        { text: 'Id', value: 'id' },
        { text: 'Name', value: 'name' },
        { text: 'Description', value: 'description' },
        { text: 'User', value: 'created_by' },
        { text: 'Creation time', value: 'created_at' },
        { text: 'Status', value: 'status' },
        {
          text: '',
          value: 'action',
          align: 'right',
          sortable: false
        }
      ]
    }
  }
})
</script>
