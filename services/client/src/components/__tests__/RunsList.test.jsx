import React from 'react';
import renderer from 'react-test-renderer';
import { shallow } from 'enzyme';

import RunsList from "../RunsList";

const runs = [
  {
    'id': 1,
    'cl': 1.0,
    'n_c': 3,
    'n_t': 3,
    'gen': 100
  },
  {
    'id': 2,
    'cl': 0.5,
    'n_c': 6,
    'n_t': 6,
    'gen': 300
  },
];

// TODO: Fix these
test('RunsList renders properly', () => {
  const wrapper = shallow(<RunsList runs={runs}/>);
  const element = wrapper.find('h4');
  expect(element.length).toBe(2);
  expect(element.get(0).props.children).toStrictEqual([1, ", ", 3, ", ", 3, ", ", 100])
});

test('RunsList renders a snapshot properly', () => {
  const tree = renderer.create(<RunsList runs={runs}/>).toJSON();
  expect(tree).toMatchSnapshot();
});