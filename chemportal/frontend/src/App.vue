<template>
  <v-app>
    <v-app-bar app dark>
      <div class="d-flex align-center">
        <v-img
          @click="$router.push('/').catch(() => {})"
          alt="Nvidia"
          class="shrink mr-2"
          style="cursor: pointer"
          contain
          src="@/assets/logo.png"
          transition="scale-transition"
        />

        <span
          @click="$router.push('/').catch(() => {})"
          style="margin-left: 24px; font-size: 250%; cursor: pointer"
        >
          Clara Drug Discovery
        </span>
      </div>

      <v-spacer></v-spacer>

      <router-link
        :to="'/wf/list'"
        style="text-decoration: none; color: inherit !important"
      >
        <span style="padding: 12px">Workflows</span>
      </router-link>
    </v-app-bar>

    <v-main>
      <router-view />
    </v-main>
  </v-app>
</template>

<script lang="ts">
import Vue from 'vue'

export default Vue.extend({
  name: 'App',

  data: () => ({
    //
  })
})
</script>
