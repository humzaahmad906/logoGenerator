export const CANVAS_ADDED = "CANVAS_ADDED";
export const OBJECT_ACTIVE = "OBJECT_ACTIVE";
export const SET_PANEL = "SET_PANEL";
export const COLOR_PANEL_PALETTES = "COLOR_PANEL_PALETTES";

export const canvasAdded = (canvas) => ({
    type: CANVAS_ADDED,
    payload: canvas,
});
export const objectActive = (object) => ({
    type: OBJECT_ACTIVE,
    payload: object,
});
export const setPanel = (panel) => ({
    type: SET_PANEL,
    payload: panel,
})
export const savePalettes = (palettes) => ({
    type: COLOR_PANEL_PALETTES,
    payload: palettes,
})