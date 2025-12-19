const path = require('path');

module.exports = function (context, options) {
  return {
    name: 'docusaurus-plugin-rag-chat',

    getClientModules() {
      return [path.resolve(__dirname, './src/client/rag-chat-injector')];
    },

    configureWebpack(config, isServer, utils) {
      return {
        resolve: {
          alias: {
            '@rag-chat': path.resolve(__dirname, '../docusaurus-components'),
          },
        },
      };
    },

    injectHtmlTags() {
      return {
        postBodyTags: [
          `<div id="rag-chat-root"></div>`,
        ],
      };
    },
  };
};