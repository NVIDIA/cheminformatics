module.exports = {
  transpileDependencies: [
    'vuetify'
  ],
  publicPath: './',
  outputDir: '../public',
  devServer: {
    host: '0.0.0.0',
    port: 8000,
     proxy: {
       '/api': {
         logLevel: 'info',
         target: 'http://127.0.0.1:5000/',
         changeOrigin: true,
         secure: false,
         pathRewrite: {
           '^/api': '/api'
         }
       }
     }
  }
}
