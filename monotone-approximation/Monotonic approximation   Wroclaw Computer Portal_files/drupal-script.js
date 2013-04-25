$( document ).ready( function() {

	// rozciaga wysokosc srodkowego kontenera
	var left_height = $('#sidebar_left').height();
	var right_height = $('#sidebar_right').height();	
	var height = left_height >= right_height ? left_height : right_height;
	
	jQuery.each(jQuery.browser, function(i) {
  		if($.browser.msie && jQuery.browser.version.substr(0,3) < "7.0") {
     		$("#content").css('height', height);
  		} else {
     		$("#content").css('min-height', height);
 		}
	});

	$(".block-user").append('<div class="sidebar_right_block_corner"></div>');
	$("#block-user-0 .sidebar_right_block_corner").remove();

        /* Add a warning about evil @hotmail.com near email input field on
           http://informatyka.wroc.pl/user/register, see trac #210.
           Only when we actually enter such e-mail address, display the warning. */

        var warning_hotmail_bad = $('<p id="warning_hotmail_bad" class="popup_warning">Uwaga: serwery pocztowe Microsoft Hotmail (<tt>@hotmail.com</tt>, <tt>@windowslive.com</tt>, i pokrewne) są <a href="http://projectile.ca/hotmail.html">niestety</a> <a href="http://www.johnberns.com/2008/02/03/windows-live-hotmail-error/">znane</a> <a href="http://www.theregister.co.uk/2007/05/01/hotmail_friendly_fire/">z cichego odrzucania maili</a> bez żadnej informacji (ani dla adresata ani dla nadawcy). Zalecamy używanie innych serwerów pocztowych.</p>');

        $('div#edit-mail-wrapper').append(warning_hotmail_bad);

        $("form#user-register input#edit-mail").change( function() {
          var email = $(this).val();
          if (wpi_is_suffix('@hotmail.com', email) ||
              wpi_is_suffix('@windowslive.com', email) ||
              wpi_is_suffix('@msn.com', email) ||
              wpi_is_suffix('@live.com', email) )
            warning_hotmail_bad.slideDown("fast");
        });
});
