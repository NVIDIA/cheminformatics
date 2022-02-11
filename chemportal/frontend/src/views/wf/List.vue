<template>
  <v-container>
    <v-toolbar
      flat
      style="background: inherit; margin-top: 24px; vertical-align: text-top"
    >
      <v-toolbar-title>Workflows</v-toolbar-title>
      <v-btn icon @click="fetchAll(true)">
        <v-icon>refresh</v-icon>
      </v-btn>

      <v-btn class="ma-2 primary"
        @click="$router.push('/wf/workflow')">
        New
        <v-icon right dark>add_circle</v-icon>
      </v-btn>

      <v-btn
        :disabled="selected.length == 0"
        color="error"
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
      v-model="selected"
      :headers="wfHeaders"
      :items="$store.state.pipeline.list"
      item-key="id"
      :search="wildSearch"
      show-select
      :options="options"
      :sort-desc="true"
      class="elevation-1"
    >
      <template v-slot:item.description="{ item }">
        {{ item.description | truncate(20, '...') }}
      </template>

      <template v-slot:item.created_at="{ item }">
        {{ $moment.utc(item.created_at).local().format('YYYY-MM-DD, h:mm:ss A') }}
      </template>
      <template v-slot:item.is_published="{ item }">
            <v-chip v-if="item.is_published" class="ma-2 green lighten-2">
              Published
            </v-chip>
            <v-chip v-else class="ma-2 red lighten-2">
              Not Published
            </v-chip>
      </template>selected

      <template v-slot:item.action="{ item }">
        <v-icon
          class="mr-2"
          @click="$router.push('/wf/workflow/' + item.id);">
          mdi-pencil
        </v-icon>
        <v-icon
          class="mr-2"
          @click="deleteItem(item)">
          mdi-delete
        </v-icon>
        <v-icon>
          mdi-cloud-upload
        </v-icon>
      </template>

    </v-data-table>
  </v-container>
</template>

<script>
import Vue from 'vue'
import CommonView from '../../mixin/CommonView'

export default Vue.extend({
  name: 'WfList',
  mixins: [CommonView],
  data: function () {
    return {
      wildSearch: '',
      selected: [],
      options: {
        itemsPerPage: 15
      },
      wfHeaders: [
        { text: 'Id', value: 'id' },
        { text: 'Name', value: 'name' },
        { text: 'Description', value: 'description' },
        { text: 'Creation time', value: 'created_at' },
        { text: 'Published', value: 'is_published' },
        {
          text: '',
          value: 'action',
          align: 'right',
          sortable: false
        }
      ]
    }
  },

  mounted () {
    this.fetchAll(false)
  },

  methods: {
    fetchAll (force) {
      this.$store.dispatch('pipeline/fetchAll', force).catch(error => {
        this.showError(null, error)
      })
    }
  }
})
</script>
