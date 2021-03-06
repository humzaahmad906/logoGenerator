import TextField from '@material-ui/core/TextField';
import React from 'react';
import { SketchPicker } from 'react-color'
import ColorSwatch from './colorswatch'
import ColorPicker from './colorpicker'

export default function CustomInput(props){
    const [colorNew, setColor] = React.useState(props.color)
    
    return(
        <div className={'d-flex flex-wrap justify-content-between'}>
            <form noValidate autoComplete="off">
                <TextField label="Text" variant="outlined" />
            </form>
            <div className={"p-3"}>
                <ColorPicker />
            </div>
            
        </div>
    )
}