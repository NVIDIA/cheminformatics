<template>
    <div style='margin: 10px'>
        <textarea
            type="text"
            rows="10"
            class="dark-input"
            :value="value"
            v-on="listeners"
            :placeholder="name"
        ></textarea>
    </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator'

@Component
export default class PrepareProtineSideBar extends Vue {
    @Prop({ type: String, default: '' })
    value!: string

    @Prop({ type: String })
    name!: string

    get listeners (): any {
      return { ...this.$listeners, input: (ev: any) => this.$emit('input', ev.target.value) }
    }
}
</script>
