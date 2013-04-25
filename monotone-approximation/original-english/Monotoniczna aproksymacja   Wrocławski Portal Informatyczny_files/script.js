/* This JS will be added to all pages under WPI theme, on Drupal and Moodle
   (possibly on phpbb too, when needed one day).

   Things that should go only into Drupal (e.g. using jQuery) must go
   to drupal-script.js, not here.
*/

/* This function is from
   http://www.netlobo.com/javascript_get_element_id.html */
function wpi_return_obj_by_id(id)
{
    if (document.getElementById)
        var returnVar = document.getElementById(id);
    else if (document.all)
        var returnVar = document.all[id];
    else if (document.layers)
        var returnVar = document.layers[id];
    return returnVar;
}

/* For article authors, see http://informatyka.wroc.pl/trac/ticket/74

   *DEPRECATED NOW*, you should use wpi_toggle_display2 (that requires a little
   different HTML markup to use, so is not compatible), see forum for authors post. */
function wpi_toggle_display(id) {
  var element = wpi_return_obj_by_id(id);
  if (element.style.display == "none")
    element.style.display=""; else
    element.style.display="none";
}

/* Note that wpi_toggle_display2 and wpi_set_display only append/remove from className,
   instead of setting/clearing className,
   this way element may have other css classes which are not touched.

   We cannot use here jQuery helpers, as this must also work in Moodle
   without jQuery stuff. */

/* Returns bool if given aClass is present in element.className. */
function wpi_hasClass(element, aClass)
{
  return (element.className.match(new RegExp('\\b' + aClass + '\\b')) != null);
}

/* Add CSS class name, if isn't already present. */
function wpi_addClass(element, aClass)
{
  if (!wpi_hasClass(element, aClass))
    element.className += ' ' + aClass;
}

/* Remove CSS class name, if present. */
function wpi_removeClass(element, aClass)
{
  /* Implementation idea from
     http://stackoverflow.com/questions/195951/change-an-elements-css-class-with-javascript */
  element.className = element.className.replace(new RegExp('\\b' + aClass + '\\b'), '');
}

function wpi_toggleClass(element, aClass)
{
  if (wpi_hasClass(element, aClass))
    wpi_removeClass(element, aClass); else
    /* We could call here wpi_addClass, but it would check wpi_hasClass again.
       Directly doing the job is a tiny performance gain. */
    element.className += ' ' + aClass;
}

function wpi_toggle_display2(id) 
{
  wpi_toggleClass(wpi_return_obj_by_id(id), "screen_display_none");
}

function wpi_set_display(id, show) 
{
  var element = wpi_return_obj_by_id(id);
  if (show)
    wpi_removeClass(element, "screen_display_none"); else
    wpi_addClass(element, "screen_display_none");
}

/* Check is "suffix" a suffix of string "text". */
function wpi_is_suffix(suffix, text)
{
  return (text.substring(text.length - suffix.length) == suffix);
}
