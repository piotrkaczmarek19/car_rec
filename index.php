<?php

$imageURL = "image.jpg";
$name = basename($imageURL);

file_put_contents("image.jpg", file_get_contents($imageURL));

chmod("image.jpg", 0777);
print "<img src='".$imageURL."'/></br>";

exec("python stream.py 2>&1", $out, $status);

$result = $out[1];

$result = str_replace(['(', ')'],'', $result);

$result = explode(",", $result);

$width = $result[2] - $result[0];
if ($result > 0)
{
	print "<h1>It's a car</h2>";
}
else
{
	print "<h1>It's NOT a car</h1>";
}
$width = $result[2] - $result[0];
if (isset($out))
{
	print "<div style='width:".$width."px; height:".$width."; border:3px solid green; position:absolute; top: ".$result[1]."; left:".$result[0]."'></div>";
}