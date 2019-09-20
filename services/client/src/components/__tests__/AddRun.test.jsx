import React from 'react';
import { shallow } from 'enzyme';
import renderer from 'react-test-renderer';

import AddRun from '../AddRun';

// TODO: Fix these
test('AddRun renders properly', () => {
  const wrapper = shallow(<AddRun/>);
  const element = wrapper.find('form');
  expect(element.find('input').length).toBe(11);
  expect(element.find('input').get(0).props.name).toBe('cl');
  expect(element.find('input').get(1).props.name).toBe('gen');
  expect(element.find('input').get(2).props.name).toBe('fix_te');
  expect(element.find('input').get(3).props.name).toBe('constrain_thickness');
  expect(element.find('input').get(4).props.name).toBe('constrain_area');
  expect(element.find('input').get(5).props.name).toBe('constrain_moment');
  expect(element.find('input').get(6).props.name).toBe('n_c');
  expect(element.find('input').get(7).props.name).toBe('n_t');
  expect(element.find('input').get(8).props.name).toBe('n_proc');
  expect(element.find('input').get(9).props.name).toBe('report');
  expect(element.find('input').get(10).props.type).toBe('submit');
});

test('AddRun renders a snapshot properly', () => {
  const tree = renderer.create(<AddRun/>).toJSON();
  expect(tree).toMatchSnapshot();
});