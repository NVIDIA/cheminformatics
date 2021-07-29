import Vue from 'vue'
import VueRouter, { RouteConfig } from 'vue-router'
import Home from '../views/Home.vue'

Vue.use(VueRouter)

const routes: Array<RouteConfig> = [
  {
    path: '/',
    name: 'Home',
    component: Home
  },
  {
    path: '/wf/list',
    name: 'WfList',
    component: () => import('../views/wf/List.vue')
  },
  {
    path: '/wf/workflow',
    name: 'Workflow',
    component: () => import('../views/wf/Workflow.vue')
  }
]

const router = new VueRouter({
  routes
})

export default router
