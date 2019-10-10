# base image
FROM node:11.12.0-alpine

# set working directory
WORKDIR /usr/src/app

# add `/usr/src/app/node_modules/.bin` to $PATH
#ENV PATH /usr/src/app/node_modules/.bin:$PATH

# install and cache app dependencies
COPY package.json /usr/src/app/package.json
COPY package-lock.json /usr/src/app/package-lock.json
RUN npm config set unsafe-perm true
RUN npm ci
RUN npm install react-scripts@2.1.8 -g --silent
RUN rm -rf node_modules/node-sass/ && npm install

# start app
CMD npm start