import React from 'react';
import PropTypes from 'prop-types';
import Popover from '@material-ui/core/Popover';
import { SketchPicker } from 'react-color'
import ColorSwatch from './colorswatch'

function SimpleDialog(props) {
  const { onClose, open } = props;
  const [colorNew, setColor] = React.useState(props.color)
  const handleColorComplete = (color) => {
      props.setSelectedValue(color.hex)
      setColor(color.hex);
  }
  const handleChange = (color, e) => {
      props.setSelectedValue(color.hex)
      setColor(color.hex);
  }
  const handleClose = () => {
    onClose(colorNew);
  };


  return (
    <Popover onClose={handleClose} aria-labelledby="simple-dialog-title" open={open}>
      <SketchPicker
            className="color-pick"
            color={colorNew}
            onChangeComplete={handleColorComplete}
            onChange={handleChange}
        />
    </Popover>
  );
}

SimpleDialog.propTypes = {
  onClose: PropTypes.func.isRequired,
  open: PropTypes.bool.isRequired,
  selectedValue: PropTypes.string.isRequired,
};

export default function ColorPicker() {
  const [open, setOpen] = React.useState(false);
  const [selectedValue, setSelectedValue] = React.useState("black");

  const handleClose = (value) => {
    setOpen(false);
    setSelectedValue(value);
  };

  return (
    <React.Fragment>
      <ColorSwatch color={selectedValue} size={"MEDIUM"} setOpen={setOpen}/>
      <SimpleDialog open={open} setSelectedValue={setSelectedValue} onClose={handleClose} />
    </React.Fragment>
  );
}