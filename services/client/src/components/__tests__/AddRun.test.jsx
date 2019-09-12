import React from 'react';
import { shallow } from 'enzyme';
import renderer from 'react-test-renderer';

import AddRun from '../AddRun';

test('AddRun renders properly', () => {
  const wrapper = shallow(<AddRun/>);
  const element = wrapper.find('form');
  expect(element.find('input').length).toBe(8);
  expect(element.find('input').get(0).props.name).toBe('cl');
  expect(element.find('input').get(1).props.name).toBe('n_c');
  expect(element.find('input').get(2).props.name).toBe('n_t');
  expect(element.find('input').get(3).props.name).toBe('b_c');
  expect(element.find('input').get(4).props.name).toBe('b_t');
  expect(element.find('input').get(5).props.name).toBe('b_te');
  expect(element.find('input').get(6).props.name).toBe('gen');
  expect(element.find('input').get(7).props.type).toBe('submit');
});

test('AddRun renders a snapshot properly', () => {
  const tree = renderer.create(<AddRun/>).toJSON();
  expect(tree).toMatchSnapshot();
});