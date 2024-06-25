const path = require('path');

module.exports = (env, argv) => {
  const isProduction = argv.mode === 'production';

  return {
    entry: {
      popup: './page/popup/popup.js',
      alarm: './page/alarm/alarm.js',
      auth: './auth.js', // auth.js 추가
      background: './background.js',
      webcam: './cam/webcam.js',
      chart: './page/chart/chart.js', // chart.js 추가
      focus: './webpage/focus/focus.js'
    },
    output: {
      filename: '[name].bundle.js',
      path: path.resolve(__dirname, 'dist'),
      publicPath: '/'
    },
    mode: isProduction ? 'production' : 'development',
    devtool: isProduction ? false : 'source-map',
    devServer: {
      static: {
        directory: path.resolve(__dirname, 'dist')
      },
      compress: true,
      port: 9000,
      open: true
    },
    module: {
      rules: [
        {
          test: /\.m?js$/,
          exclude: /node_modules/,
          use: {
            loader: 'babel-loader',
            options: {
              presets: ['@babel/preset-env']
            }
          }
        }
      ]
    }
  };
};
