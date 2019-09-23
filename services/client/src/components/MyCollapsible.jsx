import Collapsible from "react-collapsible";
import React from "react";

const MyCollapsible = (props) => {
  return (
    <Collapsible
      className="box"
      openedClassName="box"
      triggerClassName="div"
      triggerOpenedClassName="div"
      contentInnerClassName="box"
      trigger={
        <div>
          <h2 className={('titleClassName' in props) ? props.titleClassName : "title is-4"}>{props.title}</h2>
        </div>
      }>
      {props.children}
    </Collapsible>
  )
};

export default MyCollapsible;